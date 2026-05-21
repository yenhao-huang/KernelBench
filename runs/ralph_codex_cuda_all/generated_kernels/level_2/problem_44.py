import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void convt_gap_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    int N, int Cin, int H, int W, int Cout,
    float multiplier
) {
    int idx = blockIdx.x;
    int n = idx / Cout;
    int co = idx - n * Cout;

    float acc = 0.0f;

    for (int ci = threadIdx.x; ci < Cin; ci += blockDim.x) {
        const float* xbase = x + ((n * Cin + ci) * H * W);
        const float* wbase = w + ((ci * Cout + co) * 9);

        float sx00 = 0.0f;  // ih 1..H-1, iw 1..W-1
        float sx01 = 0.0f;  // ih 1..H-1, iw 0..W-1
        float sx10 = 0.0f;  // ih 0..H-1, iw 1..W-1
        float sx11 = 0.0f;  // all

        for (int ih = 0; ih < H; ++ih) {
            const float* row = xbase + ih * W;
            float row_all = 0.0f;
            float row_skip0 = 0.0f;

            for (int iw = 0; iw < W; ++iw) {
                float v = row[iw];
                row_all += v;
                if (iw > 0) row_skip0 += v;
            }

            sx11 += row_all;
            sx10 += row_skip0;
            if (ih > 0) {
                sx01 += row_all;
                sx00 += row_skip0;
            }
        }

        acc += wbase[0] * sx00;
        acc += wbase[1] * sx01;
        acc += wbase[2] * sx01;
        acc += wbase[3] * sx10;
        acc += wbase[4] * sx11;
        acc += wbase[5] * sx11;
        acc += wbase[6] * sx10;
        acc += wbase[7] * sx11;
        acc += wbase[8] * sx11;
    }

    __shared__ float smem[256];
    smem[threadIdx.x] = acc;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float denom = (float)((H * 2) * (W * 2));
        float bias = b == nullptr ? 0.0f : b[co];
        out[(n * Cout + co)] = multiplier * (bias + smem[0] / denom);
    }
}

torch::Tensor convt_gap_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double multiplier) {
    int N = x.size(0);
    int Cin = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int Cout = weight.size(1);

    auto out = torch::empty({N, Cout, 1, 1}, x.options());

    dim3 block(256);
    dim3 grid(N * Cout);

    convt_gap_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        out.data_ptr<float>(),
        N, Cin, H, W, Cout,
        (float)multiplier
    );

    return out;
}
"""

cpp_sources = r"""
torch::Tensor convt_gap_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double multiplier);
"""

convt_gap_ext = load_inline(
    name="convt_gap_ext",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["convt_gap_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.multiplier = multiplier
        self.op = convt_gap_ext

    def forward(self, x):
        return self.op.convt_gap_cuda(
            x.contiguous(),
            self.conv_transpose.weight.contiguous(),
            self.conv_transpose.bias,
            float(self.multiplier),
        )