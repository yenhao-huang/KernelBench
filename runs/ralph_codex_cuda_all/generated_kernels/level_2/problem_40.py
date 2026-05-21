import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE 16

__global__ void linear_scale_residual_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int M,
    int K,
    int N,
    float scale
) {
    __shared__ float xs[TILE][TILE + 1];
    __shared__ float ws[TILE][TILE + 1];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TILE) {
        int xk = k0 + threadIdx.x;
        int wk = k0 + threadIdx.y;

        xs[threadIdx.y][threadIdx.x] = (row < M && xk < K) ? x[row * K + xk] : 0.0f;
        ws[threadIdx.y][threadIdx.x] = (col < N && wk < K) ? weight[col * K + wk] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; ++i) {
            acc += xs[threadIdx.y][i] * ws[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        float v = acc + bias[col];
        out[row * N + col] = v * scale;
    }
}

torch::Tensor linear_scale_residual_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double scaling_factor
) {
    int M = (int)x.size(0);
    int K = (int)x.size(1);
    int N = (int)weight.size(0);

    auto out = torch::empty({M, N}, x.options());

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    float scale = (float)(1.0 + scaling_factor);

    linear_scale_residual_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        M,
        K,
        N,
        scale
    );

    return out;
}
"""

cpp_sources = r"""
torch::Tensor linear_scale_residual_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double scaling_factor
);
"""

linear_scale_residual = load_inline(
    name="linear_scale_residual_ext",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["linear_scale_residual_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.op = linear_scale_residual

    def forward(self, x):
        return self.op.linear_scale_residual_cuda(
            x.contiguous(),
            self.matmul.weight.contiguous(),
            self.matmul.bias.contiguous(),
            self.scaling_factor,
        )