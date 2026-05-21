import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# ---------- CUDA source for im2col + batched matmul with bias ----------

conv2d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// ---------- im2col kernel ----------
__global__ void im2col_kernel(const float* input, float* col_matrix,
                              int N, int C, int H, int W,
                              int kernel_h, int kernel_w,
                              int stride, int pad_h, int pad_w,
                              int dilation_h, int dilation_w,
                              int H_out, int W_out, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int M = N * H_out * W_out;
    if (idx >= M) return;

    // decode idx into (n, h_out, w_out)
    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int n = idx / (H_out * W_out);

    int base_in = n * C * H * W;
    int base_col = idx; // column index in col_matrix (shape K x M, column-major)

    for (int c = 0; c < C; ++c) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h_in = h_out * stride + kh * dilation_h - pad_h;
                int w_in = w_out * stride + kw * dilation_w - pad_w;
                int k_idx = c * kernel_h * kernel_w + kh * kernel_w + kw;
                int col_row = k_idx;
                int col_col = base_col;
                float val = 0.0f;
                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    val = input[base_in + c * H * W + h_in * W + w_in];
                }
                col_matrix[col_row * M + col_col] = val;
            }
        }
    }
}

torch::Tensor im2col_cuda(torch::Tensor input, int kernel_h, int kernel_w,
                          int stride, int pad_h, int pad_w,
                          int dilation_h, int dilation_w) {
    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    int H_out = (H + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride + 1;
    int W_out = (W + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride + 1;
    int K = C * kernel_h * kernel_w;
    int M = N * H_out * W_out;

    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto col_matrix = torch::zeros({K, M}, options);

    const int block_size = 256;
    const int num_blocks = (M + block_size - 1) / block_size;
    im2col_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), col_matrix.data_ptr<float>(),
        N, C, H, W,
        kernel_h, kernel_w, stride, pad_h, pad_w,
        dilation_h, dilation_w,
        H_out, W_out, K);
    return col_matrix;
}

// ---------- tiled matrix multiply (with optional bias) ----------
#define TILE_SIZE 16

__global__ void matmul_bias_kernel(const float* A, const float* B, float* C,
                                   const float* bias, int M, int K, int N) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile from A (M x K)
        int a_row = row;
        int a_col = t * TILE_SIZE + threadIdx.x;
        if (a_row < M && a_col < K)
            tile_A[threadIdx.y][threadIdx.x] = A[a_row * K + a_col];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile from B (K x N)
        int b_row = t * TILE_SIZE + threadIdx.y;
        int b_col = col;
        if (b_row < K && b_col < N)
            tile_B[threadIdx.y][threadIdx.x] = B[b_row * N + b_col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        float out = sum;
        if (bias != nullptr) {
            out += bias[row];
        }
        C[row * N + col] = out;
    }
}

torch::Tensor matmul_bias_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor bias) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    auto C = torch::zeros({M, N}, options);

    dim3 block_dim(TILE_SIZE, TILE_SIZE);
    dim3 grid_dim((N + TILE_SIZE - 1) / TILE_SIZE,
                  (M + TILE_SIZE - 1) / TILE_SIZE);

    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;

    matmul_bias_kernel<<<grid_dim, block_dim>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        bias_ptr, M, K, N);

    return C;
}
"""

conv2d_cpp_source = """
torch::Tensor im2col_cuda(torch::Tensor input, int kernel_h, int kernel_w,
                          int stride, int pad_h, int pad_w,
                          int dilation_h, int dilation_w);
torch::Tensor matmul_bias_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor bias);
"""

# Compile the inline CUDA code
custom_conv2d = load_inline(
    name="custom_conv2d",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_cuda_source,
    functions=["im2col_cuda", "matmul_bias_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# ---------- Optimised Model ----------
class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1),
                 bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_h, self.kernel_w = kernel_size
        self.stride = stride
        self.pad_h, self.pad_w = padding
        self.dilation_h, self.dilation_w = dilation
        self.bias_flag = bias

        # Weight: out_channels x (in_channels * kernel_h * kernel_w)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels * self.kernel_h * self.kernel_w)
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

        self.custom_conv = custom_conv2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        # Ensure input is contiguous and float32
        x = x.contiguous()
        # Step 1: im2col
        col = self.custom_conv.im2col_cuda(
            x, self.kernel_h, self.kernel_w,
            self.stride, self.pad_h, self.pad_w,
            self.dilation_h, self.dilation_w
        )
        # col shape: (C*kernel_h*kernel_w, N*H_out*W_out)
        # Step 2: matrix multiply weight (out_channels x K) with col (K x M) + bias
        out_2d = self.custom_conv.matmul_bias_cuda(self.weight, col, self.bias)
        # out_2d shape: (out_channels, N*H_out*W_out)
        # Step 3: reshape to (N, out_channels, H_out, W_out)
        H_out = (H + 2 * self.pad_h - self.dilation_h * (self.kernel_h - 1) - 1) // self.stride + 1
        W_out = (W + 2 * self.pad_w - self.dilation_w * (self.kernel_w - 1) - 1) // self.stride + 1
        out = out_2d.view(N, self.out_channels, H_out, W_out)
        return out