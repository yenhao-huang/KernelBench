import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

group_norm_cpp_source = """
torch::Tensor group_norm_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int64_t num_groups, double eps);
"""

group_norm_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void group_norm_stats_kernel(
    const float* __restrict__ x,
    float* __restrict__ mean,
    float* __restrict__ invstd,
    int N,
    int C,
    int G,
    int inner,
    float eps
) {
    int ng = blockIdx.x;
    int n = ng / G;
    int g = ng - n * G;
    int group_channels = C / G;
    int count = group_channels * inner;
    int base = (n * C + g * group_channels) * inner;

    float local_sum = 0.0f;
    float local_sumsq = 0.0f;

    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        float v = x[base + i];
        local_sum += v;
        local_sumsq += v * v;
    }

    __shared__ float s_sum[256];
    __shared__ float s_sumsq[256];

    s_sum[threadIdx.x] = local_sum;
    s_sumsq[threadIdx.x] = local_sumsq;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
            s_sumsq[threadIdx.x] += s_sumsq[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float m = s_sum[0] / (float)count;
        float var = s_sumsq[0] / (float)count - m * m;
        var = fmaxf(var, 0.0f);
        mean[ng] = m;
        invstd[ng] = rsqrtf(var + eps);
    }
}

__global__ void group_norm_apply_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ mean,
    const float* __restrict__ invstd,
    float* __restrict__ out,
    int64_t total,
    int C,
    int G,
    int inner
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    int group_channels = C / G;

    for (; idx < total; idx += stride) {
        int s = idx % inner;
        int64_t t = idx / inner;
        int c = t % C;
        int n = t / C;
        int g = c / group_channels;
        int mg = n * G + g;

        float v = (x[idx] - mean[mg]) * invstd[mg];
        out[idx] = v * weight[c] + bias[c];
    }
}

torch::Tensor group_norm_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int64_t num_groups, double eps) {
    int N = (int)x.size(0);
    int C = (int)x.size(1);
    int G = (int)num_groups;
    int inner = (int)(x.numel() / (N * C));
    int64_t total = x.numel();

    auto out = torch::empty_like(x);
    auto stats_opts = x.options();
    auto mean = torch::empty({N, G}, stats_opts);
    auto invstd = torch::empty({N, G}, stats_opts);

    const int stats_threads = 256;
    const int apply_threads = 256;
    int stats_blocks = N * G;
    int apply_blocks = (int)((total + apply_threads - 1) / apply_threads);
    if (apply_blocks > 65535) apply_blocks = 65535;

    group_norm_stats_kernel<<<stats_blocks, stats_threads>>>(
        x.data_ptr<float>(),
        mean.data_ptr<float>(),
        invstd.data_ptr<float>(),
        N, C, G, inner, (float)eps
    );

    group_norm_apply_kernel<<<apply_blocks, apply_threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        mean.data_ptr<float>(),
        invstd.data_ptr<float>(),
        out.data_ptr<float>(),
        total, C, G, inner
    );

    return out;
}
"""

_group_norm_mod = load_inline(
    name="custom_group_norm_fp32_kernelbench",
    cpp_sources=group_norm_cpp_source,
    cuda_sources=group_norm_cuda_source,
    functions=["group_norm_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    def __init__(self, num_features: int, num_groups: int):
        super().__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _group_norm_mod.group_norm_cuda(
            x.contiguous(),
            self.weight,
            self.bias,
            self.num_groups,
            1e-5,
        )