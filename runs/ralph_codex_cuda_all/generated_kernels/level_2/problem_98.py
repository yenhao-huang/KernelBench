import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math_constants.h>

__global__ void collapse_weight_bias_kernel(
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ cw,
    float* __restrict__ cb,
    int out_features,
    int in_features,
    int pool_k,
    int pooled
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = pooled * in_features;
    if (idx < total) {
        int j = idx / in_features;
        int k = idx - j * in_features;
        float s = 0.0f;
        int base = j * pool_k;
        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            if (r < pool_k) {
                s += w[(base + r) * in_features + k];
            }
        }
        cw[idx] = s / (float)pool_k;
    }

    int bidx = idx - total;
    if (bidx >= 0 && bidx < pooled) {
        float s = 0.0f;
        int base = bidx * pool_k;
        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            if (r < pool_k) {
                s += bias[base + r];
            }
        }
        cb[bidx] = s / (float)pool_k;
    }
}

__device__ __forceinline__ float gelu_approx(float x) {
    const float c = 0.7978845608028654f;
    const float k = 0.044715f;
    float u = c * (x + k * x * x * x);
    return 0.5f * x * (1.0f + tanhf(u));
}

__global__ void fused_linear_pool_gelu_scale_max_kernel(
    const float* __restrict__ x,
    const float* __restrict__ cw,
    const float* __restrict__ cb,
    float* __restrict__ out,
    int batch,
    int in_features,
    int pooled,
    float scale
) {
    int b = blockIdx.x;
    int j = blockIdx.y;
    int tid = threadIdx.x;

    float acc = 0.0f;
    const float* xb = x + b * in_features;
    const float* wj = cw + j * in_features;

    for (int k = tid; k < in_features; k += blockDim.x) {
        acc += xb[k] * wj[k];
    }

    __shared__ float smem[256];
    smem[tid] = acc;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    __shared__ float vals[512];
    if (tid == 0) {
        float v = gelu_approx(smem[0] + cb[j]) * scale;
        vals[j] = v;
    }
    __syncthreads();

    if (j == 0) {
        float m = -CUDART_INF_F;
        for (int t = tid; t < pooled; t += blockDim.x) {
            m = fmaxf(m, vals[t]);
        }
        smem[tid] = m;
        __syncthreads();

        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
            }
            __syncthreads();
        }

        if (tid == 0) {
            out[b] = smem[0];
        }
    }
}

torch::Tensor fused_forward_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int64_t pool_k, double scale) {
    int batch = (int)x.size(0);
    int in_features = (int)x.size(1);
    int out_features = (int)weight.size(0);
    int pooled = out_features / (int)pool_k;

    auto cw = torch::empty({pooled, in_features}, x.options());
    auto cb = torch::empty({pooled}, x.options());
    auto out = torch::empty({batch}, x.options());

    int total = pooled * in_features + pooled;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    collapse_weight_bias_kernel<<<blocks, threads>>>(
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        cw.data_ptr<float>(),
        cb.data_ptr<float>(),
        out_features,
        in_features,
        (int)pool_k,
        pooled
    );

    dim3 grid(batch, pooled);
    fused_linear_pool_gelu_scale_max_kernel<<<grid, threads>>>(
        x.data_ptr<float>(),
        cw.data_ptr<float>(),
        cb.data_ptr<float>(),
        out.data_ptr<float>(),
        batch,
        in_features,
        pooled,
        (float)scale
    );

    return out;
}
"""

cpp_sources = r"""
torch::Tensor fused_forward_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int64_t pool_k, double scale);
"""

fused_ops = load_inline(
    name="matmul_avgpool_gelu_scale_max_fused",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["fused_forward_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.pool_kernel_size = int(pool_kernel_size)
        self.scale_factor = float(scale_factor)
        self.fused_ops = fused_ops

    def forward(self, x):
        return self.fused_ops.fused_forward_cuda(
            x.contiguous(),
            self.matmul.weight.contiguous(),
            self.matmul.bias.contiguous(),
            self.pool_kernel_size,
            self.scale_factor,
        )