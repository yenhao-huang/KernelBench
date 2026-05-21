import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA source for fused matmul + bias + Swish + scale
fused_matmul_swish_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_M 32
#define TILE_N 32
#define TILE_K 32

__global__ void fused_matmul_swish_scale_kernel(
    const float* __restrict__ A,
    const float* __restrict__ W,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K,
    float scale)
{
    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;

    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Ws[TILE_N][TILE_K];

    for (int k_block = 0; k_block < K; k_block += TILE_K) {
        // Load A tile
        if (row < M && (k_block + threadIdx.x) < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + k_block + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load W tile
        if (col < N && (k_block + threadIdx.y) < K) {
            Ws[threadIdx.x][threadIdx.y] = W[col * K + k_block + threadIdx.y];
        } else {
            Ws[threadIdx.x][threadIdx.y] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            sum += As[threadIdx.y][k] * Ws[threadIdx.x][k];
        }

        __syncthreads();
    }

    // Add bias
    float y = sum + bias[col];

    // Swish activation: y * sigmoid(y)
    float sig = 1.0f / (1.0f + expf(-y));
    y = y * sig * scale;

    C[row * N + col] = y;
}

torch::Tensor fused_matmul_swish_scale_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaling_factor)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");

    int M = x.size(0);    // batch_size
    int K = x.size(1);    // in_features
    int N = weight.size(0); // out_features

    TORCH_CHECK(weight.size(1) == K, "weight shape mismatch");
    TORCH_CHECK(bias.size(0) == N, "bias shape mismatch");

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto out = torch::empty({M, N}, options);

    dim3 block(TILE_N, TILE_M);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    fused_matmul_swish_scale_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        M, N, K,
        scaling_factor
    );

    return out;
}
"""

fused_matmul_swish_scale_cpp_source = """
torch::Tensor fused_matmul_swish_scale_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaling_factor);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_matmul_swish_scale",
    cpp_sources=fused_matmul_swish_scale_cpp_source,
    cuda_sources=fused_matmul_swish_scale_source,
    functions=["fused_matmul_swish_scale_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        # Keep nn.Linear for proper weight/bias initialization
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.fused_op = fused_op

    def forward(self, x):
        return self.fused_op.fused_matmul_swish_scale_cuda(
            x,
            self.matmul.weight,
            self.matmul.bias,
            self.scaling_factor
        )