import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BM 16
#define BN 16
#define BK 32

__global__ void linear_relu_div_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y,
    int M,
    int K,
    int N,
    float divisor
) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BN][BK];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * BM + ty;
    int col = blockIdx.x * BN + tx;
    int tid = ty * blockDim.x + tx;

    float acc = 0.0f;

    for (int kt = 0; kt < K; kt += BK) {
        for (int idx = tid; idx < BM * BK; idx += BM * BN) {
            int r = idx / BK;
            int k = idx - r * BK;
            int gr = blockIdx.y * BM + r;
            int gk = kt + k;
            As[r][k] = (gr < M && gk < K) ? x[gr * K + gk] : 0.0f;
        }

        for (int idx = tid; idx < BN * BK; idx += BM * BN) {
            int c = idx / BK;
            int k = idx - c * BK;
            int gc = blockIdx.x * BN + c;
            int gk = kt + k;
            Bs[c][k] = (gc < N && gk < K) ? w[gc * K + gk] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            acc += As[ty][k] * Bs[tx][k];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        acc += b[col];
        acc = acc > 0.0f ? acc : 0.0f;
        y[row * N + col] = acc / divisor;
    }
}

torch::Tensor linear_relu_div_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b, double divisor) {
    int M = (int)x.size(0);
    int K = (int)x.size(1);
    int N = (int)w.size(0);

    auto y = torch::empty({M, N}, x.options());

    dim3 block(BN, BM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    linear_relu_div_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        b.data_ptr<float>(),
        y.data_ptr<float>(),
        M,
        K,
        N,
        (float)divisor
    );

    return y;
}
"""

cpp_sources = r"""
torch::Tensor linear_relu_div_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b, double divisor);
"""

linear_relu_div_ext = load_inline(
    name="linear_relu_div_ext",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["linear_relu_div_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.divisor = divisor
        self.op = linear_relu_div_ext

    def forward(self, x):
        return self.op.linear_relu_div_cuda(x, self.linear.weight, self.linear.bias, self.divisor)