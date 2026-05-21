import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define BM 8
#define BN 64
#define BK 32

__global__ void linear_swish_scale_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    int M,
    int K,
    int N,
    float scale
) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BN][BK];

    int row_local = threadIdx.y;
    int col_group = threadIdx.x;
    int row = blockIdx.y * BM + row_local;
    int col0 = blockIdx.x * BN + col_group * 2;
    int col1 = col0 + 1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    for (int kt = 0; kt < K; kt += BK) {
        for (int i = tid; i < BM * BK; i += blockDim.x * blockDim.y) {
            int r = i / BK;
            int k = i - r * BK;
            int gr = blockIdx.y * BM + r;
            int gk = kt + k;
            As[r][k] = (gr < M && gk < K) ? x[gr * K + gk] : 0.0f;
        }

        for (int i = tid; i < BN * BK; i += blockDim.x * blockDim.y) {
            int c = i / BK;
            int k = i - c * BK;
            int gc = blockIdx.x * BN + c;
            int gk = kt + k;
            Bs[c][k] = (gc < N && gk < K) ? w[gc * K + gk] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a = As[row_local][k];
            if (col0 < N) acc0 += a * Bs[col_group * 2][k];
            if (col1 < N) acc1 += a * Bs[col_group * 2 + 1][k];
        }

        __syncthreads();
    }

    if (row < M && col0 < N) {
        float v = acc0 + b[col0];
        out[row * N + col0] = (v / (1.0f + expf(-v))) * scale;
    }
    if (row < M && col1 < N) {
        float v = acc1 + b[col1];
        out[row * N + col1] = (v / (1.0f + expf(-v))) * scale;
    }
}

torch::Tensor linear_swish_scale_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b, double scale) {
    int M = (int)x.size(0);
    int K = (int)x.size(1);
    int N = (int)w.size(0);

    auto out = torch::empty({M, N}, x.options());

    dim3 block(32, BM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    linear_swish_scale_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        M,
        K,
        N,
        (float)scale
    );

    return out;
}
"""

cpp_sources = r"""
torch::Tensor linear_swish_scale_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b, double scale);
"""

linear_swish_scale_ext = load_inline(
    name="linear_swish_scale_ext",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["linear_swish_scale_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.ext = linear_swish_scale_ext

    def forward(self, x):
        return self.ext.linear_swish_scale_cuda(
            x.contiguous(),
            self.matmul.weight.contiguous(),
            self.matmul.bias.contiguous(),
            float(self.scaling_factor),
        )