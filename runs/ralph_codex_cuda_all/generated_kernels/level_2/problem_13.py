import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void mean_bias_softmax_tanh_scale_kernel(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int B, int C, int D, int H, int W,
    float scale
) {
    int pos = blockIdx.x;
    int hw = H * W;
    int b = pos / hw;
    int rem = pos - b * hw;
    int h = rem / W;
    int w = rem - h * W;

    extern __shared__ float vals[];

    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float sum = 0.0f;
        int base = (((b * C + c) * D) * H + h) * W + w;
        int plane = H * W;
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            sum += x[base + d * plane];
        }
        vals[c] = sum / (float)D + bias[c];
    }
    __syncthreads();

    float local_max = -FLT_MAX;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        local_max = fmaxf(local_max, vals[c]);
    }

    __shared__ float red[256];
    red[threadIdx.x] = local_max;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        }
        __syncthreads();
    }
    float maxv = red[0];

    float local_sum = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float e = expf(vals[c] - maxv);
        vals[c] = e;
        local_sum += e;
    }

    red[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            red[threadIdx.x] += red[threadIdx.x + s];
        }
        __syncthreads();
    }
    float inv_sum = 1.0f / red[0];

    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float soft = vals[c] * inv_sum;
        out[((b * C + c) * H + h) * W + w] = tanhf(soft) * scale;
    }
}

torch::Tensor fused_post_cuda(torch::Tensor x, torch::Tensor bias, double scale) {
    int B = x.size(0);
    int C = x.size(1);
    int D = x.size(2);
    int H = x.size(3);
    int W = x.size(4);

    auto out = torch::empty({B, C, 1, H, W}, x.options());

    int threads = 256;
    int blocks = B * H * W;
    size_t shmem = C * sizeof(float);

    mean_bias_softmax_tanh_scale_kernel<<<blocks, threads, shmem>>>(
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        B, C, D, H, W,
        (float)scale
    );

    return out;
}
"""

cpp_sources = r"""
torch::Tensor fused_post_cuda(torch::Tensor x, torch::Tensor bias, double scale);
"""

fused_post = load_inline(
    name="kb_fused_convtranspose3d_post_v1",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["fused_post_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor
        self.fused_post = fused_post

    def forward(self, x):
        x = self.conv_transpose(x)
        return self.fused_post.fused_post_cuda(x.contiguous(), self.bias.contiguous(), self.scaling_factor)