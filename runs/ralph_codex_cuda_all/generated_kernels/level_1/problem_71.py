import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose2d_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K, int stride, int padding, int output_padding,
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

    int oc_per_group = C_out / groups;
    int ic_per_group = C_in / groups;
    int g = oc / oc_per_group;
    int ic_start = g * ic_per_group;
    int ic_end = ic_start + ic_per_group;
    int ocg = oc - g * oc_per_group;

    float acc = has_bias ? bias[oc] : 0.0f;

    for (int ic = ic_start; ic < ic_end; ++ic) {
        for (int kh = 0; kh < K; ++kh) {
            int ih_unstrided = oh + padding - kh;
            if (ih_unstrided % stride != 0) continue;
            int ih = ih_unstrided / stride;
            if (ih < 0 || ih >= H_in) continue;

            for (int kw = 0; kw < K; ++kw) {
                int iw_unstrided = ow + padding - kw;
                if (iw_unstrided % stride != 0) continue;
                int iw = iw_unstrided / stride;
                if (iw < 0 || iw >= W_in) continue;

                int x_idx = ((n * C_in + ic) * H_in + ih) * W_in + iw;
                int w_idx = ((ic * oc_per_group + ocg) * K + kh) * K + kw;
                acc += x[x_idx] * w[w_idx];
            }
        }
    }

    out[idx] = acc;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int output_padding,
    int groups,
    bool has_bias
) {
    int N = x.size(0);
    int C_in = x.size(1);
    int H_in = x.size(2);
    int W_in = x.size(3);
    int K = weight.size(2);
    int C_out = weight.size(1) * groups;

    int H_out = (H_in - 1) * stride - 2 * padding + K + output_padding;
    int W_out = (W_in - 1) * stride - 2 * padding + K + output_padding;

    auto out = torch::empty({N, C_out, H_out, W_out}, x.options());

    int total = N * C_out * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    conv_transpose2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        has_bias ? bias.data_ptr<float>() : nullptr,
        out.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K, stride, padding, output_padding,
        groups, has_bias ? 1 : 0
    );

    return out;
}
"""

conv_transpose2d_cpp = "torch::Tensor conv_transpose2d_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int stride, int padding, int output_padding, int groups, bool has_bias);"

conv_transpose2d_ext = load_inline(
    name="conv_transpose2d_ext_kernelbench",
    cpp_sources=conv_transpose2d_cpp,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
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
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels // groups, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = in_channels * kernel_size * kernel_size / groups
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias if self.bias is not None else torch.empty(0, device=x.device, dtype=x.dtype)
        return conv_transpose2d_ext.conv_transpose2d_cuda(
            x,
            self.weight,
            bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.bias is not None,
        )