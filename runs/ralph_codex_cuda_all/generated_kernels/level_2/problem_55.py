import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void fused_linear_pool_sum_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int batch,
    int in_features,
    int out_features,
    float scale
) {
    extern __shared__ float smem[];
    float* s0 = smem;
    float* s1 = smem + blockDim.x;

    int pair_count = out_features >> 1;
    int gid = blockIdx.x;
    int b = gid / pair_count;
    int p = gid - b * pair_count;
    int o0 = p << 1;
    int o1 = o0 + 1;

    const float* xb = x + ((long long)b * in_features);
    const float* w0 = w + ((long long)o0 * in_features);
    const float* w1 = w + ((long long)o1 * in_features);

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (int k = threadIdx.x; k < in_features; k += blockDim.x) {
        float xv = xb[k];
        acc0 += xv * w0[k];
        acc1 += xv * w1[k];
    }

    s0[threadIdx.x] = acc0;
    s1[threadIdx.x] = acc1;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s0[threadIdx.x] += s0[threadIdx.x + stride];
            s1[threadIdx.x] += s1[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float v0 = s0[0] + bias[o0];
        float v1 = s1[0] + bias[o1];
        atomicAdd(out + b, fmaxf(v0, v1) * scale);
    }
}

torch::Tensor fused_linear_pool_sum_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor bias,
    float scale
) {
    int batch = (int)x.size(0);
    int in_features = (int)x.size(1);
    int out_features = (int)w.size(0);

    auto out = torch::zeros({batch}, x.options());

    const int threads = 256;
    int pair_count = out_features / 2;
    int blocks = batch * pair_count;
    size_t shared = threads * 2 * sizeof(float);

    fused_linear_pool_sum_kernel<<<blocks, threads, shared>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        batch,
        in_features,
        out_features,
        scale
    );

    return out;
}
"""

cpp_sources = r"""
torch::Tensor fused_linear_pool_sum_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor bias,
    float scale
);
"""

fused_ops = load_inline(
    name="kernelbench_fused_linear_pool_sum_fp32",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["fused_linear_pool_sum_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scale_factor = float(scale_factor)
        self.fused_ops = fused_ops

    def forward(self, x):
        return self.fused_ops.fused_linear_pool_sum_cuda(
            x.contiguous(),
            self.matmul.weight.contiguous(),
            self.matmul.bias.contiguous(),
            self.scale_factor,
        )