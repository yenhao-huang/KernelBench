import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define GROUP_SIZE 64
#define THREADS 256

__global__ void fused_linear_swish_bias_groupnorm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ linear_bias,
    const float* __restrict__ add_bias,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    int B,
    int K,
    int C,
    int G
) {
    int bg = blockIdx.x;
    int b = bg / G;
    int g = bg - b * G;
    int tid = threadIdx.x;
    int c0 = g * GROUP_SIZE;

    __shared__ float partial[GROUP_SIZE][THREADS];
    __shared__ float vals[GROUP_SIZE];
    __shared__ float mean_s;
    __shared__ float invstd_s;

    for (int ci = 0; ci < GROUP_SIZE; ++ci) {
        int c = c0 + ci;
        float acc = 0.0f;
        for (int k = tid; k < K; k += THREADS) {
            acc += x[b * K + k] * w[c * K + k];
        }
        partial[ci][tid] = acc;
    }
    __syncthreads();

    if (tid < GROUP_SIZE) {
        float sum = 0.0f;
        #pragma unroll
        for (int t = 0; t < THREADS; ++t) {
            sum += partial[tid][t];
        }
        int c = c0 + tid;
        float z = sum + linear_bias[c];
        float swish = z / (1.0f + expf(-z));
        float v = swish + add_bias[c];
        vals[tid] = v;
    }
    __syncthreads();

    if (tid == 0) {
        float m = 0.0f;
        #pragma unroll
        for (int i = 0; i < GROUP_SIZE; ++i) {
            m += vals[i];
        }
        m *= 1.0f / (float)GROUP_SIZE;

        float var = 0.0f;
        #pragma unroll
        for (int i = 0; i < GROUP_SIZE; ++i) {
            float d = vals[i] - m;
            var += d * d;
        }
        var *= 1.0f / (float)GROUP_SIZE;

        mean_s = m;
        invstd_s = rsqrtf(var + 1.0e-5f);
    }
    __syncthreads();

    if (tid < GROUP_SIZE) {
        int c = c0 + tid;
        out[b * C + c] = (vals[tid] - mean_s) * invstd_s * gamma[c] + beta[c];
    }
}

torch::Tensor fused_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor linear_bias,
    torch::Tensor add_bias,
    torch::Tensor gamma,
    torch::Tensor beta,
    int64_t num_groups
) {
    int B = (int)x.size(0);
    int K = (int)x.size(1);
    int C = (int)weight.size(0);
    int G = (int)num_groups;

    auto out = torch::empty({B, C}, x.options());

    dim3 block(THREADS);
    dim3 grid(B * G);

    fused_linear_swish_bias_groupnorm_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        linear_bias.data_ptr<float>(),
        add_bias.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        out.data_ptr<float>(),
        B, K, C, G
    );

    return out;
}
"""

cpp_sources = r"""
torch::Tensor fused_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor linear_bias,
    torch::Tensor add_bias,
    torch::Tensor gamma,
    torch::Tensor beta,
    int64_t num_groups
);
"""

fused_ext = load_inline(
    name="kb_fused_linear_swish_bias_groupnorm_fp32",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["fused_forward_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.num_groups = num_groups
        self.fused_ext = fused_ext

    def forward(self, x):
        return self.fused_ext.fused_forward_cuda(
            x,
            self.matmul.weight,
            self.matmul.bias,
            self.bias,
            self.group_norm.weight,
            self.group_norm.bias,
            self.num_groups,
        )