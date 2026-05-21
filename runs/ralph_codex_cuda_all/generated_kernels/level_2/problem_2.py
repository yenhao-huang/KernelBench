import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_fused_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ conv_b,
    const float* __restrict__ add_b,
    float* __restrict__ out,
    int N, int IC, int IH, int IW,
    int OC, int KH, int KW,
    int OH, int OW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    float scale
) {
    long long total = (long long)N * OC * OH * OW;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;

    for (; idx < total; idx += (long long)blockDim.x * gridDim.x) {
        int ow = idx % OW;
        long long t = idx / OW;
        int oh = t % OH;
        t /= OH;
        int oc = t % OC;
        int n = t / OC;

        float acc = conv_b ? conv_b[oc] : 0.0f;

        for (int ic = 0; ic < IC; ++ic) {
            for (int kh = 0; kh < KH; ++kh) {
                int ih_unscaled = oh + pad_h - kh;
                if (ih_unscaled < 0 || ih_unscaled % stride_h != 0) continue;
                int ih = ih_unscaled / stride_h;
                if (ih < 0 || ih >= IH) continue;

                for (int kw = 0; kw < KW; ++kw) {
                    int iw_unscaled = ow + pad_w - kw;
                    if (iw_unscaled < 0 || iw_unscaled % stride_w != 0) continue;
                    int iw = iw_unscaled / stride_w;
                    if (iw < 0 || iw >= IW) continue;

                    long long x_idx = ((long long)n * IC + ic) * IH * IW + ih * IW + iw;
                    long long w_idx = ((long long)ic * OC + oc) * KH * KW + kh * KW + kw;
                    acc += x[x_idx] * w[w_idx];
                }
            }
        }

        acc += add_b[oc];
        acc = fminf(fmaxf(acc, 0.0f), 1.0f);
        acc = fminf(fmaxf(acc * scale, 0.0f), 1.0f);
        out[idx] = acc / scale;
    }
}

torch::Tensor conv_transpose2d_fused_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor add_bias,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int out_pad_h,
    int out_pad_w,
    float scale
) {
    int N = x.size(0);
    int IC = x.size(1);
    int IH = x.size(2);
    int IW = x.size(3);
    int OC = weight.size(1);
    int KH = weight.size(2);
    int KW = weight.size(3);

    int OH = (IH - 1) * stride_h - 2 * pad_h + KH + out_pad_h;
    int OW = (IW - 1) * stride_w - 2 * pad_w + KW + out_pad_w;

    auto out = torch::empty({N, OC, OH, OW}, x.options());

    long long total = (long long)N * OC * OH * OW;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;

    conv_transpose2d_fused_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.numel() ? conv_bias.data_ptr<float>() : nullptr,
        add_bias.data_ptr<float>(),
        out.data_ptr<float>(),
        N, IC, IH, IW, OC, KH, KW, OH, OW,
        stride_h, stride_w, pad_h, pad_w, scale
    );

    return out;
}
"""

cpp_sources = r"""
torch::Tensor conv_transpose2d_fused_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor add_bias,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int out_pad_h,
    int out_pad_w,
    float scale
);
"""

conv_transpose2d_fused = load_inline(
    name="conv_transpose2d_fused_ext",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["conv_transpose2d_fused_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
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
        self.scaling_factor = float(scaling_factor)

        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding)
        self.op = conv_transpose2d_fused

    def forward(self, x):
        return self.op.conv_transpose2d_fused_cuda(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.bias,
            int(self.stride[0]),
            int(self.stride[1]),
            int(self.padding[0]),
            int(self.padding[1]),
            int(self.output_padding[0]),
            int(self.output_padding[1]),
            self.scaling_factor,
        )