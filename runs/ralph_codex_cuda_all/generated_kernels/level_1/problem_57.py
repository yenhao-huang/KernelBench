import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose2d_cpp_source = """
torch::Tensor conv_transpose2d_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
                                    int stride_h, int stride_w, int pad_h, int pad_w,
                                    int out_pad_h, int out_pad_w, int groups, bool has_bias);
"""

conv_transpose2d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y,
    int N, int C_in, int H, int W,
    int C_out, int K_h, int K_w,
    int H_out, int W_out,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int groups, int has_bias
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

    int c_in_per_group = C_in / groups;
    int c_out_per_group = C_out / groups;
    int g = oc / c_out_per_group;
    int ic_start = g * c_in_per_group;
    int ic_end = ic_start + c_in_per_group;
    int ocg = oc - g * c_out_per_group;

    float acc = has_bias ? b[oc] : 0.0f;

    for (int ic = ic_start; ic < ic_end; ++ic) {
        const float* x_base = x + ((n * C_in + ic) * H * W);
        const float* w_base = w + ((ic * c_out_per_group + ocg) * K_h * K_w);

        for (int kh = 0; kh < K_h; ++kh) {
            int ih_num = oh + pad_h - kh;
            if (ih_num % stride_h != 0) continue;
            int ih = ih_num / stride_h;
            if ((unsigned)ih >= (unsigned)H) continue;

            for (int kw = 0; kw < K_w; ++kw) {
                int iw_num = ow + pad_w - kw;
                if (iw_num % stride_w != 0) continue;
                int iw = iw_num / stride_w;
                if ((unsigned)iw >= (unsigned)W) continue;

                acc += x_base[ih * W + iw] * w_base[kh * K_w + kw];
            }
        }
    }

    y[idx] = acc;
}

torch::Tensor conv_transpose2d_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
                                    int stride_h, int stride_w, int pad_h, int pad_w,
                                    int out_pad_h, int out_pad_w, int groups, bool has_bias) {
    int N = x.size(0);
    int C_in = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    int K_h = weight.size(2);
    int K_w = weight.size(3);
    int C_out = weight.size(1) * groups;

    int H_out = (H - 1) * stride_h - 2 * pad_h + K_h + out_pad_h;
    int W_out = (W - 1) * stride_w - 2 * pad_w + K_w + out_pad_w;

    auto y = torch::empty({N, C_out, H_out, W_out}, x.options());

    int total = N * C_out * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    conv_transpose2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        has_bias ? bias.data_ptr<float>() : nullptr,
        y.data_ptr<float>(),
        N, C_in, H, W,
        C_out, K_h, K_w,
        H_out, W_out,
        stride_h, stride_w,
        pad_h, pad_w,
        groups, has_bias ? 1 : 0
    );

    return y;
}
"""

conv_transpose2d_ext = load_inline(
    name="conv_transpose2d_inline_kernelbench",
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
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kh, kw = kernel_size
        else:
            kh = kw = kernel_size
        if isinstance(stride, tuple):
            self.stride_h, self.stride_w = stride
        else:
            self.stride_h = self.stride_w = stride
        if isinstance(padding, tuple):
            self.pad_h, self.pad_w = padding
        else:
            self.pad_h = self.pad_w = padding
        if isinstance(output_padding, tuple):
            self.out_pad_h, self.out_pad_w = output_padding
        else:
            self.out_pad_h = self.out_pad_w = output_padding

        self.groups = groups
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels // groups, kh, kw))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias if self.bias is not None else torch.empty(0, device=x.device, dtype=x.dtype)
        return conv_transpose2d_ext.conv_transpose2d_cuda(
            x.contiguous(),
            self.weight.contiguous(),
            bias,
            self.stride_h,
            self.stride_w,
            self.pad_h,
            self.pad_w,
            self.out_pad_h,
            self.out_pad_w,
            self.groups,
            self.bias is not None,
        )