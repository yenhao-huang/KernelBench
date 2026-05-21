import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE 16

__global__ void gemm_mul_leaky_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    int M,
    int K,
    int N,
    float multiplier,
    float negative_slope
) {
    __shared__ float xs[TILE][TILE + 1];
    __shared__ float ws[TILE][TILE + 1];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    for (int t = 0; t < K; t += TILE) {
        int kx = t + threadIdx.x;
        int ky = t + threadIdx.y;

        xs[threadIdx.y][threadIdx.x] = (row < M && kx < K) ? x[row * K + kx] : 0.0f;
        ws[threadIdx.y][threadIdx.x] = (col < N && ky < K) ? w[col * K + ky] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; ++i) {
            acc += xs[threadIdx.y][i] * ws[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        float v = (acc + b[col]) * multiplier;
        out[row * N + col] = v >= 0.0f ? v : v * negative_slope;
    }
}

torch::Tensor gemm_mul_leaky_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    double multiplier,
    double negative_slope
) {
    int M = x.size(0);
    int K = x.size(1);
    int N = w.size(0);

    auto out = torch::empty({M, N}, x.options());

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    gemm_mul_leaky_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        M,
        K,
        N,
        static_cast<float>(multiplier),
        static_cast<float>(negative_slope)
    );

    return out;
}
"""

cpp_sources = r"""
torch::Tensor gemm_mul_leaky_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    double multiplier,
    double negative_slope
);
"""

gemm_mul_leaky_ext = load_inline(
    name="gemm_mul_leaky_ext",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["gemm_mul_leaky_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.multiplier = float(multiplier)
        self.negative_slope = float(negative_slope)
        self.op = gemm_mul_leaky_ext

    def forward(self, x):
        return self.op.gemm_mul_leaky_cuda(
            x,
            self.gemm.weight,
            self.gemm.bias,
            self.multiplier,
            self.negative_slope,
        )