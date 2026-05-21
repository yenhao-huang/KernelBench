import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BM 16
#define BN 16
#define BK 32

__global__ void linear_bias_relu_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int M,
    int K,
    int N
) {
    __shared__ float xs[BM][BK + 1];
    __shared__ float ws[BN][BK + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * BM + ty;
    int col = blockIdx.x * BN + tx;

    float acc = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int i = tx; i < BK; i += BN) {
            int k = k0 + i;
            xs[ty][i] = (row < M && k < K) ? x[row * K + k] : 0.0f;
        }

        for (int i = ty; i < BK; i += BM) {
            int k = k0 + i;
            ws[tx][i] = (col < N && k < K) ? w[col * K + k] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            acc += xs[ty][k] * ws[tx][k];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        float v = acc + bias[col];
        out[row * N + col] = v > 0.0f ? v : 0.0f;
    }
}

torch::Tensor linear_bias_relu_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor bias) {
    int M = (int)x.size(0);
    int K = (int)x.size(1);
    int N = (int)w.size(0);

    auto out = torch::empty({M, N}, x.options());

    dim3 block(BN, BM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    linear_bias_relu_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        M,
        K,
        N
    );

    return out;
}
"""

cpp_sources = "torch::Tensor linear_bias_relu_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor bias);"

linear_bias_relu_ext = load_inline(
    name="linear_bias_relu_ext",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["linear_bias_relu_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        return linear_bias_relu_ext.linear_bias_relu_cuda(x, self.gemm.weight, self.bias)