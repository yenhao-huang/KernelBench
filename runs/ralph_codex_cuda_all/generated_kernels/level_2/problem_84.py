import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BM 16
#define BN 16
#define BK 32

__global__ void linear_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y,
    int M, int K, int N
) {
    __shared__ float xs[BM][BK + 1];
    __shared__ float ws[BK][BN + 1];

    int row = blockIdx.y * BM + threadIdx.y;
    int col = blockIdx.x * BN + threadIdx.x;

    float acc = 0.0f;

    for (int t = 0; t < K; t += BK) {
        int x_col = t + threadIdx.x;
        int w_row = t + threadIdx.y;

        if (threadIdx.x < BK && row < M && x_col < K) {
            xs[threadIdx.y][threadIdx.x] = x[row * K + x_col];
        }

        if (threadIdx.y < BK && w_row < K && col < N) {
            ws[threadIdx.y][threadIdx.x] = w[col * K + w_row];
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            if (t + k < K) {
                acc += xs[threadIdx.y][k] * ws[k][threadIdx.x];
            }
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        y[row * N + col] = acc + b[col];
    }
}

torch::Tensor linear_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
    int M = x.size(0);
    int K = x.size(1);
    int N = w.size(0);

    auto y = torch::empty({M, N}, x.options());

    dim3 block(BN, BM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    linear_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        b.data_ptr<float>(),
        y.data_ptr<float>(),
        M, K, N
    );

    return y;
}
"""

cpp_sources = r"""
torch::Tensor linear_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b);
"""

linear_ext = load_inline(
    name="kernelbench_linear_bn_scale_softmax_ext",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["linear_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = linear_ext.linear_cuda(x.contiguous(), self.gemm.weight.contiguous(), self.gemm.bias.contiguous())
        x = self.bn(x)
        x = self.scale * x
        x = self.softmax(x)
        return x