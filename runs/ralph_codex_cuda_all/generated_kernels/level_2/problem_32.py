import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void conv_scale_channel_min_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    int N, int C, int H, int W,
    int O, int K, int OH, int OW,
    float scale
) {
    const int lanes_per_pixel = 128;
    const int pixels_per_block = 2;
    const int lane = threadIdx.x & (lanes_per_pixel - 1);
    const int group = threadIdx.x >> 7;
    const int pix = blockIdx.x * pixels_per_block + group;
    const int total = N * OH * OW;

    __shared__ float vals[pixels_per_block][lanes_per_pixel];

    float v = FLT_MAX;
    if (pix < total && lane < O) {
        int tmp = pix;
        const int ox = tmp % OW;
        tmp /= OW;
        const int oy = tmp % OH;
        const int n = tmp / OH;

        float acc = b ? b[lane] : 0.0f;

        for (int c = 0; c < C; ++c) {
            for (int ky = 0; ky < K; ++ky) {
                const int iy = oy + ky;
                for (int kx = 0; kx < K; ++kx) {
                    const int ix = ox + kx;
                    const int x_idx = ((n * C + c) * H + iy) * W + ix;
                    const int w_idx = ((lane * C + c) * K + ky) * K + kx;
                    acc = fmaf(x[x_idx], w[w_idx], acc);
                }
            }
        }
        v = acc * scale;
    }

    vals[group][lane] = v;
    __syncthreads();

    if (lane < 64) vals[group][lane] = fminf(vals[group][lane], vals[group][lane + 64]);
    __syncthreads();
    if (lane < 32) {
        float m = vals[group][lane];
        m = fminf(m, vals[group][lane + 32]);
        m = fminf(m, __shfl_down_sync(0xffffffff, m, 16));
        m = fminf(m, __shfl_down_sync(0xffffffff, m, 8));
        m = fminf(m, __shfl_down_sync(0xffffffff, m, 4));
        m = fminf(m, __shfl_down_sync(0xffffffff, m, 2));
        m = fminf(m, __shfl_down_sync(0xffffffff, m, 1));
        if (lane == 0 && pix < total) {
            out[pix] = m;
        }
    }
}

torch::Tensor conv_scale_channel_min_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double scale
) {
    const int N = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int O = weight.size(0);
    const int K = weight.size(2);
    const int OH = H - K + 1;
    const int OW = W - K + 1;

    auto out = torch::empty({N, 1, OH, OW}, x.options());

    const int pixels_per_block = 2;
    const int threads = 256;
    const int total = N * OH * OW;
    const int blocks = (total + pixels_per_block - 1) / pixels_per_block;

    conv_scale_channel_min_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() && bias.numel() > 0 ? bias.data_ptr<float>() : nullptr,
        out.data_ptr<float>(),
        N, C, H, W, O, K, OH, OW,
        static_cast<float>(scale)
    );

    return out;
}
"""

cpp_sources = r"""
torch::Tensor conv_scale_channel_min_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double scale
);
"""

conv_scale_channel_min_ext = load_inline(
    name="conv_scale_channel_min_ext",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["conv_scale_channel_min_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = float(scale_factor)
        self.op = conv_scale_channel_min_ext

    def forward(self, x):
        return self.op.conv_scale_channel_min_cuda(
            x.contiguous(),
            self.conv.weight.contiguous(),
            self.conv.bias.contiguous(),
            self.scale_factor,
        )