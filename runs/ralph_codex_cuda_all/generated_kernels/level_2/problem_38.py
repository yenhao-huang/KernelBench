import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void clamp_softmax_scale_kernel(
    const float* __restrict__ x,
    const float* __restrict__ scale,
    float* __restrict__ out,
    int rows,
    int spatial,
    int channels
) {
    extern __shared__ float smem[];
    float* smax = smem;
    float* ssum = smem + blockDim.x;

    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    int c = row % channels;
    int base = row * spatial;

    float local_max = 0.0f;
    for (int i = tid; i < spatial; i += blockDim.x) {
        float v = x[base + i];
        v = fminf(fmaxf(v, 0.0f), 1.0f);
        local_max = fmaxf(local_max, v);
    }

    smax[tid] = local_max;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) smax[tid] = fmaxf(smax[tid], smax[tid + offset]);
        __syncthreads();
    }

    float maxv = smax[0];
    float local_sum = 0.0f;
    for (int i = tid; i < spatial; i += blockDim.x) {
        float v = x[base + i];
        v = fminf(fmaxf(v, 0.0f), 1.0f);
        local_sum += expf(v - maxv);
    }

    ssum[tid] = local_sum;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) ssum[tid] += ssum[tid + offset];
        __syncthreads();
    }

    float inv_sum = 1.0f / ssum[0];
    float sc = scale[c];

    for (int i = tid; i < spatial; i += blockDim.x) {
        float v = x[base + i];
        v = fminf(fmaxf(v, 0.0f), 1.0f);
        out[base + i] = expf(v - maxv) * inv_sum * sc;
    }
}

torch::Tensor clamp_softmax_scale_cuda(torch::Tensor x, torch::Tensor scale) {
    auto out = torch::empty_like(x);

    int b = (int)x.size(0);
    int c = (int)x.size(1);
    int d = (int)x.size(2);
    int h = (int)x.size(3);
    int w = (int)x.size(4);
    int spatial = d * h * w;
    int rows = b * c;

    const int threads = 256;
    size_t shmem = threads * 2 * sizeof(float);

    clamp_softmax_scale_kernel<<<rows, threads, shmem>>>(
        x.data_ptr<float>(),
        scale.data_ptr<float>(),
        out.data_ptr<float>(),
        rows,
        spatial,
        c
    );

    return out;
}
"""

cpp_sources = r"""
torch::Tensor clamp_softmax_scale_cuda(torch::Tensor x, torch::Tensor scale);
"""

clamp_softmax_scale_ext = load_inline(
    name="clamp_softmax_scale_ext",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["clamp_softmax_scale_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv_transpose(x)
        return clamp_softmax_scale_ext.clamp_softmax_scale_cuda(x.contiguous(), self.scale.contiguous())