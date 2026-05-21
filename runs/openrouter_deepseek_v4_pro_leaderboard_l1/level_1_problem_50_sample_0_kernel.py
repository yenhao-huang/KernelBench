import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

# Define the CUDA source for a tiled matrix multiplication (GEMM) kernel
custom_gemm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void gemm_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    float sum = 0.0f;
    for (int ks = 0; ks < K; ks += BLOCK_SIZE) {
        // Load A tile
        if (row < M && (ks + tx) < K)
            As[ty][tx] = A[row * K + (ks + tx)];
        else
            As[ty][tx] = 0.0f;

        // Load B tile
        if (col < N && (ks + ty) < K)
            Bs[ty][tx] = B[(ks + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor gemm_cuda(torch::Tensor A, torch::Tensor B) {
    // A: (M, K), B: (K, N) -> C: (M, N)
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dims mismatch");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kFloat32).device(A.device()));

    // Make sure inputs are contiguous row-major
    A = A.contiguous();
    B = B.contiguous();

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    gemm_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    return C;
}
"""

custom_gemm_cpp_source = (
    "torch::Tensor gemm_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for GEMM
custom_gemm_module = load_inline(
    name="custom_gemm",
    cpp_sources=custom_gemm_cpp_source,
    cuda_sources=custom_gemm_source,
    functions=["gemm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.out_channels = 96
        self.in_channels = 3
        self.kernel_size = 11
        self.stride = 4
        self.padding = 2

        # Weight and bias parameters replacing the Conv2d layer
        self.weight = nn.Parameter(torch.empty(96, 3, 11, 11))
        self.bias = nn.Parameter(torch.empty(96))
        self.reset_parameters()

        self.gemm = custom_gemm_module

    def reset_parameters(self):
        # Kaiming initialization mimicking nn.Conv2d default
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        batch = x.shape[0]
        # im2col via unfold
        patches = F.unfold(x, kernel_size=11, stride=4, padding=2)  # (B, C*kh*kw, L)

        # Prepare matrices for GEMM: A (M, K), B (K, N)
        K = self.weight.size(1) * self.weight.size(2) * self.weight.size(3)  # C*kH*kW
        N = self.weight.size(0)  # out_channels

        L = patches.size(2)  # number of spatial outputs (55*55)
        M = batch * L

        # Reshape patches to (M, K) – each row is a pixel’s receptive field
        A = patches.permute(0, 2, 1).reshape(M, K).contiguous()

        # Weight matrix: (N, K) -> transpose to (K, N)
        B = self.weight.reshape(N, K).t().contiguous()

        # Custom GEMM
        C = self.gemm.gemm_cuda(A, B)  # (M, N)

        # Add bias (broadcast across M)
        C += self.bias.view(1, N)

        # Reshape back to (B, C, H_out, W_out)
        H_out = W_out = 55  # computed from input size 224, kernel 11, stride 4, pad 2
        C = C.reshape(batch, L, N).permute(0, 2, 1).reshape(batch, N, H_out, W_out)

        return C