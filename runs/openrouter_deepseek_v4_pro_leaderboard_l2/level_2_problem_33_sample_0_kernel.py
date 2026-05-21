import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source code for fused GEMM + bias + scale
fused_gemm_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE 32

__global__ void fused_gemm_scale_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    const float* __restrict__ scale,
    float* __restrict__ C,
    int M, int N, int K) {

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE + ty;
    int col = bx * TILE + tx;

    float sum = 0.0f;

    for (int k = 0; k < K; k += TILE) {
        // Load A tile
        if (row < M && (k + tx) < K)
            As[ty][tx] = A[row * K + (k + tx)];
        else
            As[ty][tx] = 0.0f;

        // Load B tile (B is N x K, we need B[col][k + ty])
        if (col < N && (k + ty) < K)
            Bs[tx][ty] = B[col * K + (k + ty)];
        else
            Bs[tx][ty] = 0.0f;

        __syncthreads();

        // Accumulate dot product
        for (int kk = 0; kk < TILE; ++kk)
            sum += As[ty][kk] * Bs[tx][kk];

        __syncthreads();
    }

    // Apply bias and scale
    if (row < M && col < N) {
        float val = sum + bias[col];
        val = val * scale[col];
        C[row * N + col] = val;
    }
}

torch::Tensor fused_gemm_scale_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor scale) {

    const int M = input.size(0);
    const int K = input.size(1);
    const int N = weight.size(0);

    auto output = torch::empty({M, N}, input.options());

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    fused_gemm_scale_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        scale.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K);

    return output;
}
"""

fused_gemm_scale_cpp_source = "torch::Tensor fused_gemm_scale_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor scale);"

# Compile the inline CUDA code
fused_gemm_scale = load_inline(
    name="fused_gemm_scale",
    cpp_sources=fused_gemm_scale_cpp_source,
    cuda_sources=fused_gemm_scale_source,
    functions=["fused_gemm_scale_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        self.fused_gemm_scale = fused_gemm_scale

    def forward(self, x):
        # Fused GEMM + bias + scale
        x = self.fused_gemm_scale.fused_gemm_scale_cuda(
            x, self.gemm.weight, self.gemm.bias, self.scale
        )
        # Batch normalization
        x = self.bn(x)
        return x