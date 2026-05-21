import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose2d_cpp_source = """
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int out_pad_h,
    int out_pad_w,
    int dilation_h,
    int dilation_w,
    int groups
);
"""

conv_transpose2d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y,
    int N,
    int C_in,
    int H_in,
    int W_in,
    int C_out,
    int H_out,
    int W_out,
    int K_h,
    int K_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int groups,
    int has_bias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H_out * W_out;
    if (idx >= total) return;

    int ow = idx % W_out;
    int t = idx / W_out;
    int oh = t % H_out;
    t /= H_out;
    int oc = t % C_out;
    int n = t / C_out;

    int c_out_per_group = C_out / groups;
    int c_in_per_group = C_in / groups;
    int g = oc / c_out_per_group;
    int ic_start = g * c_in_per_group;
    int ic_end = ic_start + c_in_per_group;
    int ocg = oc - g * c_out_per_group;

    float acc = has_bias ? b[oc] : 0.0f;

    #pragma unroll
    for (int kh = 0; kh < K_h; ++kh) {
        int ih_unstrided = oh + pad_h - kh * dilation_h;
        if (ih_unstrided % stride_h != 0) continue;
        int ih = ih_unstrided / stride_h;
        if (ih < 0 || ih >= H_in) continue;

        #pragma unroll
        for (int kw = 0; kw < K_w; ++kw) {
            int iw_unstrided = ow + pad_w - kw * dilation_w;
            if (iw_unstrided % stride_w != 0) continue;
            int iw = iw_unstrided / stride_w;
            if (iw < 0 || iw >= W_in) continue;

            for (int ic = ic_start; ic < ic_end; ++ic) {
                float xv = x[((n * C_in + ic) * H_in + ih) * W_in + iw];
                float wv = w[(((ic * c_out_per_group + ocg) * K_h + kh) * K_w + kw)];
                acc += xv * wv;
            }
        }
    }

    y[idx] = acc;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int out_pad_h,
    int out_pad_w,
    int dilation_h,
    int dilation_w,
    int groups
) {
    int N = x.size(0);
    int C_in = x.size(1);
    int H_in = x.size(2);
    int W_in = x.size(3);
    int K_h = weight.size(2);
    int K_w = weight.size(3);
    int C_out = weight.size(1) * groups;

    int H_out = (H_in - 1) * stride_h - 2 * pad_h + dilation_h * (K_h - 1) + out_pad_h + 1;
    int W_out = (W_in - 1) * stride_w - 2 * pad_w + dilation_w * (K_w - 1) + out_pad_w + 1;

    auto y = torch::empty({N, C_out, H_out, W_out}, x.options());

    int total = N * C_out * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    conv_transpose2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.numel() ? bias.data_ptr<float>() : nullptr,
        y.data_ptr<float>(),
        N, C_in, H_in, W_in, C_out, H_out, W_out, K_h, K_w,
        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
        groups, bias.numel() ? 1 : 0
    );

    return y;
}
"""

conv_transpose2d_ext = load_inline(
    name="conv_transpose2d_inline_ext",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_cuda_source,
    functions=["conv_transpose2d_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        output_padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.conv_transpose2d.bias
        if bias is None:
            bias = torch.empty(0, device=x.device, dtype=x.dtype)

        return conv_transpose2d_ext.conv_transpose2d_cuda(
            x.contiguous(),
            self.conv_transpose2d.weight.contiguous(),
            bias,
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1],
            self.output_padding[0],
            self.output_padding[1],
            self.dilation[0],
            self.dilation[1],
            self.groups,
        )