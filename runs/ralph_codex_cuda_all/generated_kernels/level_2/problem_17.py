import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void instancenorm_div_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int groups,
    int spatial,
    float inv_divide_by,
    float eps
) {
    int g = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float shared[];
    float* s_sum = shared;
    float* s_sq = shared + blockDim.x;

    const float* base = x + ((long long)g * spatial);
    float sum = 0.0f;
    float sq = 0.0f;

    for (int i = tid; i < spatial; i += blockDim.x) {
        float v = base[i];
        sum += v;
        sq += v * v;
    }

    s_sum[tid] = sum;
    s_sq[tid] = sq;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sq[tid] += s_sq[tid + stride];
        }
        __syncthreads();
    }

    float mean = s_sum[0] / spatial;
    float var = s_sq[0] / spatial - mean * mean;
    float scale = rsqrtf(var + eps) * inv_divide_by;

    float* out_base = out + ((long long)g * spatial);
    for (int i = tid; i < spatial; i += blockDim.x) {
        out_base[i] = (base[i] - mean) * scale;
    }
}

torch::Tensor instancenorm_div_cuda(torch::Tensor x, double divide_by, double eps) {
    auto out = torch::empty_like(x);

    int n = x.size(0);
    int c = x.size(1);
    int h = x.size(2);
    int w = x.size(3);
    int groups = n * c;
    int spatial = h * w;

    const int threads = 256;
    size_t shared_bytes = threads * 2 * sizeof(float);

    instancenorm_div_kernel<<<groups, threads, shared_bytes>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        groups,
        spatial,
        1.0f / static_cast<float>(divide_by),
        static_cast<float>(eps)
    );

    return out;
}
"""

cpp_sources = "torch::Tensor instancenorm_div_cuda(torch::Tensor x, double divide_by, double eps);"

instancenorm_div_ext = load_inline(
    name="instancenorm_div_ext",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["instancenorm_div_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divide_by = divide_by
        self.instancenorm_div = instancenorm_div_ext

    def forward(self, x):
        x = self.conv(x)
        return self.instancenorm_div.instancenorm_div_cuda(x, float(self.divide_by), 1e-5)