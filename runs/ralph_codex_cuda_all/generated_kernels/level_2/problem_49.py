import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

softmax_sigmoid_cpp_source = """
torch::Tensor softmax_sigmoid_cuda(torch::Tensor x);
"""

softmax_sigmoid_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void softmax_sigmoid_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int total_positions,
    int channels,
    int spatial_size
) {
    int pos = blockIdx.x;
    if (pos >= total_positions) return;

    int tid = threadIdx.x;
    int n = pos / spatial_size;
    int s = pos - n * spatial_size;
    int base = n * channels * spatial_size + s;

    float local_max = -FLT_MAX;
    for (int c = tid; c < channels; c += blockDim.x) {
        float v = x[base + c * spatial_size];
        local_max = fmaxf(local_max, v);
    }

    __shared__ float smem[256];
    smem[tid] = local_max;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            smem[tid] = fmaxf(smem[tid], smem[tid + offset]);
        }
        __syncthreads();
    }

    float max_v = smem[0];
    float local_sum = 0.0f;

    for (int c = tid; c < channels; c += blockDim.x) {
        local_sum += expf(x[base + c * spatial_size] - max_v);
    }

    smem[tid] = local_sum;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            smem[tid] += smem[tid + offset];
        }
        __syncthreads();
    }

    float inv_sum = 1.0f / smem[0];

    for (int c = tid; c < channels; c += blockDim.x) {
        float softmax_v = expf(x[base + c * spatial_size] - max_v) * inv_sum;
        out[base + c * spatial_size] = 1.0f / (1.0f + expf(-softmax_v));
    }
}

torch::Tensor softmax_sigmoid_cuda(torch::Tensor x) {
    auto out = torch::empty_like(x);

    int n = (int)x.size(0);
    int c = (int)x.size(1);
    int d = (int)x.size(2);
    int h = (int)x.size(3);
    int w = (int)x.size(4);

    int spatial_size = d * h * w;
    int total_positions = n * spatial_size;

    const int threads = 256;
    softmax_sigmoid_kernel<<<total_positions, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        total_positions,
        c,
        spatial_size
    );

    return out;
}
"""

softmax_sigmoid_ext = load_inline(
    name="softmax_sigmoid_ext_kb_ct3d",
    cpp_sources=softmax_sigmoid_cpp_source,
    cuda_sources=softmax_sigmoid_cuda_source,
    functions=["softmax_sigmoid_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=bias,
        )
        self.softmax_sigmoid = softmax_sigmoid_ext

    def forward(self, x):
        x = self.conv_transpose(x)
        return self.softmax_sigmoid.softmax_sigmoid_cuda(x)