import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define TILE 16

__device__ __forceinline__ float mish_once(float x) {
    float sp;
    if (x > 20.0f) {
        sp = x;
    } else if (x < -20.0f) {
        sp = expf(x);
    } else {
        sp = log1pf(expf(x));
    }
    return x * tanhf(sp);
}

__global__ void linear_mish2_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    int M,
    int K,
    int N
) {
    __shared__ float xs[TILE][TILE];
    __shared__ float ws[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    for (int t = 0; t < K; t += TILE) {
        int x_col = t + threadIdx.x;
        int w_k = t + threadIdx.y;

        xs[threadIdx.y][threadIdx.x] = (row < M && x_col < K) ? x[row * K + x_col] : 0.0f;
        ws[threadIdx.y][threadIdx.x] = (col < N && w_k < K) ? w[col * K + w_k] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; ++i) {
            acc += xs[threadIdx.y][i] * ws[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        float v = acc + b[col];
        v = mish_once(v);
        v = mish_once(v);
        out[row * N + col] = v;
    }
}

torch::Tensor linear_mish2_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
    int M = (int)x.size(0);
    int K = (int)x.size(1);
    int N = (int)w.size(0);

    auto out = torch::empty({M, N}, x.options());

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    linear_mish2_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        M,
        K,
        N
    );

    return out;
}
"""

cpp_sources = "torch::Tensor linear_mish2_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b);"

linear_mish2_ext = load_inline(
    name="linear_mish2_ext",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["linear_mish2_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        fan_in = in_features
        bound = 1 / (fan_in ** 0.5)
        nn.init.uniform_(self.bias, -bound, bound)
        self.op = linear_mish2_ext

    def forward(self, x):
        return self.op.linear_mish2_cuda(x, self.weight, self.bias)