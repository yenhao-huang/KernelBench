import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define TILE 16
#define GN_THREADS 128

__device__ __forceinline__ float mish_hardtanh(float x) {
    x = fminf(1.0f, fmaxf(-1.0f, x));
    return x * tanhf(log1pf(expf(x)));
}

__global__ void linear_bias_act_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ linear_bias,
    const float* __restrict__ extra_bias,
    float* __restrict__ out,
    int N,
    int K,
    int C
) {
    __shared__ float xs[TILE][TILE];
    __shared__ float ws[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    for (int t = 0; t < K; t += TILE) {
        int x_col = t + threadIdx.x;
        int w_k = t + threadIdx.y;

        xs[threadIdx.y][threadIdx.x] = (row < N && x_col < K) ? x[row * K + x_col] : 0.0f;
        ws[threadIdx.y][threadIdx.x] = (col < C && w_k < K) ? w[col * K + w_k] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; ++i) {
            acc += xs[threadIdx.y][i] * ws[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < C) {
        float v = acc + linear_bias[col] + extra_bias[col];
        out[row * C + col] = mish_hardtanh(v);
    }
}

__global__ void groupnorm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    int N,
    int C,
    int G,
    float eps
) {
    int ng = blockIdx.x;
    int n = ng / G;
    int g = ng - n * G;
    int group_size = C / G;
    int start = g * group_size;

    __shared__ float smem[GN_THREADS];

    float sum = 0.0f;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        sum += x[n * C + start + i];
    }

    smem[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }

    float mean = smem[0] / (float)group_size;

    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float d = x[n * C + start + i] - mean;
        var_sum += d * d;
    }

    smem[threadIdx.x] = var_sum;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }

    float inv_std = rsqrtf(smem[0] / (float)group_size + eps);

    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        int c = start + i;
        float v = (x[n * C + c] - mean) * inv_std;
        out[n * C + c] = v * gamma[c] + beta[c];
    }
}

torch::Tensor fused_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor linear_bias,
    torch::Tensor extra_bias,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    int64_t num_groups,
    double eps
) {
    int N = (int)x.size(0);
    int K = (int)x.size(1);
    int C = (int)weight.size(0);

    auto tmp = torch::empty({N, C}, x.options());
    auto out = torch::empty({N, C}, x.options());

    dim3 block(TILE, TILE);
    dim3 grid((C + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    linear_bias_act_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        linear_bias.data_ptr<float>(),
        extra_bias.data_ptr<float>(),
        tmp.data_ptr<float>(),
        N, K, C
    );

    groupnorm_kernel<<<N * (int)num_groups, GN_THREADS>>>(
        tmp.data_ptr<float>(),
        gn_weight.data_ptr<float>(),
        gn_bias.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, (int)num_groups, (float)eps
    );

    return out;
}
"""

cpp_sources = r"""
torch::Tensor fused_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor linear_bias,
    torch::Tensor extra_bias,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    int64_t num_groups,
    double eps
);
"""

fused_ops = load_inline(
    name="kb_gemm_bias_hardtanh_mish_groupnorm",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["fused_forward_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)
        self.num_groups = num_groups
        self.eps = self.groupnorm.eps
        self.fused_ops = fused_ops

    def forward(self, x):
        return self.fused_ops.fused_forward_cuda(
            x.contiguous(),
            self.gemm.weight.contiguous(),
            self.gemm.bias.contiguous(),
            self.bias.contiguous(),
            self.groupnorm.weight.contiguous(),
            self.groupnorm.bias.contiguous(),
            self.num_groups,
            self.eps,
        )