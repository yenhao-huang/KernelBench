import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#define TILE 16

__global__ void linear_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y,
    int B, int K, int O
) {
    __shared__ float xs[TILE][TILE];
    __shared__ float ws[TILE][TILE];

    int n = blockIdx.y * TILE + threadIdx.y;
    int o = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    for (int t = 0; t < K; t += TILE) {
        int kx = t + threadIdx.x;
        int ky = t + threadIdx.y;

        xs[threadIdx.y][threadIdx.x] = (n < B && kx < K) ? x[n * K + kx] : 0.0f;
        ws[threadIdx.y][threadIdx.x] = (o < O && ky < K) ? w[o * K + ky] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; ++i) {
            acc += xs[threadIdx.y][i] * ws[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (n < B && o < O) {
        y[n * O + o] = acc + b[o];
    }
}

__global__ void groupnorm_min_kernel(
    const float* __restrict__ y,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ mins,
    int B, int O, int G, float eps
) {
    int n = blockIdx.x;
    int tid = threadIdx.x;
    int group_size = O / G;

    __shared__ float block_min[256];

    float local_min = FLT_MAX;

    for (int g = tid; g < G; g += blockDim.x) {
        int base = n * O + g * group_size;

        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            sum += y[base + i];
        }

        float mean = sum / (float)group_size;

        float var_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            float d = y[base + i] - mean;
            var_sum += d * d;
        }

        float inv_std = rsqrtf(var_sum / (float)group_size + eps);

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int c = g * group_size + i;
            float v = (y[base + i] - mean) * inv_std;
            v = v * gamma[c] + beta[c];
            local_min = fminf(local_min, v);
        }
    }

    block_min[tid] = local_min;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            block_min[tid] = fminf(block_min[tid], block_min[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        mins[n] = block_min[0];
    }
}

__global__ void add_bias_kernel(
    const float* __restrict__ mins,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int B, int O
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * O;

    if (idx < total) {
        int n = idx % B;
        int o = idx / B;
        out[idx] = mins[n] + bias[o];
    }
}

torch::Tensor fused_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor linear_bias,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    torch::Tensor final_bias,
    int64_t groups,
    double eps
) {
    int B = (int)x.size(0);
    int K = (int)x.size(1);
    int O = (int)weight.size(0);

    auto y = torch::empty({B, O}, x.options());
    auto mins = torch::empty({B}, x.options());
    auto out = torch::empty({1, O, B, 1}, x.options());

    dim3 block(TILE, TILE);
    dim3 grid((O + TILE - 1) / TILE, (B + TILE - 1) / TILE);
    linear_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        linear_bias.data_ptr<float>(),
        y.data_ptr<float>(),
        B, K, O
    );

    groupnorm_min_kernel<<<B, 256>>>(
        y.data_ptr<float>(),
        gn_weight.data_ptr<float>(),
        gn_bias.data_ptr<float>(),
        mins.data_ptr<float>(),
        B, O, (int)groups, (float)eps
    );

    int total = B * O;
    add_bias_kernel<<<(total + 255) / 256, 256>>>(
        mins.data_ptr<float>(),
        final_bias.data_ptr<float>(),
        out.data_ptr<float>(),
        B, O
    );

    return out;
}
"""

cpp_sources = r"""
torch::Tensor fused_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor linear_bias,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    torch::Tensor final_bias,
    int64_t groups,
    double eps
);
"""

fused_ext = load_inline(
    name="kb_gemm_groupnorm_min_bias_ext",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["fused_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.num_groups = num_groups

    def forward(self, x):
        return fused_ext.fused_cuda(
            x.contiguous(),
            self.gemm.weight.contiguous(),
            self.gemm.bias.contiguous(),
            self.group_norm.weight.contiguous(),
            self.group_norm.bias.contiguous(),
            self.bias.contiguous().view(-1),
            self.num_groups,
            self.group_norm.eps,
        )