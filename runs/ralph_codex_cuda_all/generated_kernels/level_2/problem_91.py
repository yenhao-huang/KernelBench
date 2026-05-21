import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void softmax_bias_scale_sigmoid_kernel(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int N,
    int C,
    int H,
    int W,
    float scale
) {
    int site = blockIdx.x;
    int hw = H * W;
    int n = site / hw;
    int s = site - n * hw;
    int tid = threadIdx.x;

    extern __shared__ float smem[];
    float v = -FLT_MAX;

    if (tid < C) {
        v = x[((n * C + tid) * H * W) + s];
    }
    smem[tid] = v;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            float other = smem[tid + offset];
            smem[tid] = smem[tid] > other ? smem[tid] : other;
        }
        __syncthreads();
    }

    float maxv = smem[0];
    float e = 0.0f;
    if (tid < C) {
        e = __expf(x[((n * C + tid) * H * W) + s] - maxv);
    }
    smem[tid] = e;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            smem[tid] += smem[tid + offset];
        }
        __syncthreads();
    }

    if (tid < C) {
        float y = e / smem[0];
        y = (y + bias[tid]) * scale;
        out[((n * C + tid) * H * W) + s] = 1.0f / (1.0f + __expf(-y));
    }
}

torch::Tensor softmax_bias_scale_sigmoid_cuda(torch::Tensor x, torch::Tensor bias, double scale) {
    auto out = torch::empty_like(x);
    int N = (int)x.size(0);
    int C = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);

    int threads = 1;
    while (threads < C) threads <<= 1;
    if (threads < 32) threads = 32;

    int blocks = N * H * W;
    size_t shared = threads * sizeof(float);

    softmax_bias_scale_sigmoid_kernel<<<blocks, threads, shared>>>(
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W,
        (float)scale
    );

    return out;
}
"""

cpp_sources = r"""
torch::Tensor softmax_bias_scale_sigmoid_cuda(torch::Tensor x, torch::Tensor bias, double scale);
"""

_softmax_post = load_inline(
    name="kb_convtranspose_softmax_post_fused",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["softmax_bias_scale_sigmoid_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self._softmax_post = _softmax_post

    def forward(self, x):
        x = self.conv_transpose(x)
        return self._softmax_post.softmax_bias_scale_sigmoid_cuda(x, self.bias, self.scaling_factor)