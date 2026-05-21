import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cpp_sources = """
torch::Tensor fused_linear_acts_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor add_value);
"""

cuda_sources = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define BM 16
#define BN 16
#define BK 32

__device__ __forceinline__ float gelu_exact(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
}

__device__ __forceinline__ float sigmoidf_fast(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void fused_linear_acts_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ add_value,
    float* __restrict__ out,
    int M,
    int K,
    int N
) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * BM + ty;
    int col = blockIdx.x * BN + tx;

    float acc = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BK) {
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int kk = k0 + i * 16 + tx;
            As[ty][i * 16 + tx] = (row < M && kk < K) ? x[row * K + kk] : 0.0f;
        }

        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int kk = k0 + i * 16 + ty;
            Bs[i * 16 + ty][tx] = (col < N && kk < K) ? weight[col * K + kk] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        float v = acc + bias[col] + add_value[col];

        v = v * sigmoidf_fast(v);
        v = tanhf(v);
        v = gelu_exact(v);
        v = fminf(1.0f, fmaxf(-1.0f, v));

        out[row * N + col] = v;
    }
}

torch::Tensor fused_linear_acts_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor add_value) {
    int M = (int)x.size(0);
    int K = (int)x.size(1);
    int N = (int)weight.size(0);

    auto out = torch::empty({M, N}, x.options());

    dim3 threads(BN, BM);
    dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);

    fused_linear_acts_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        add_value.data_ptr<float>(),
        out.data_ptr<float>(),
        M,
        K,
        N
    );

    return out;
}
"""

fused_linear_acts = load_inline(
    name="fused_linear_acts_kernelbench",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["fused_linear_acts_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, add_value_shape):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape))

    def forward(self, x):
        return fused_linear_acts.fused_linear_acts_cuda(
            x.contiguous(),
            self.matmul.weight.contiguous(),
            self.matmul.bias.contiguous(),
            self.add_value.contiguous(),
        )