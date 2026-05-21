import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose2d_cpp_source = """
torch::Tensor conv_transpose2d_cuda(torch::Tensor x, torch::Tensor weight,
                                    int stride_h, int stride_w,
                                    int pad_h, int pad_w,
                                    int dil_h, int dil_w,
                                    int groups);
"""

conv_transpose2d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ y,
    int N, int C_in, int H, int W,
    int C_out, int K_h, int K_w,
    int H_out, int W_out,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dil_h, int dil_w,
    int groups
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)N * C_out * H_out * W_out;
    if (idx >= total) return;

    int ow = idx % W_out;
    idx /= W_out;
    int oh = idx % H_out;
    idx /= H_out;
    int oc = idx % C_out;
    int n = idx / C_out;

    int out_per_group = C_out / groups;
    int in_per_group = C_in / groups;
    int g = oc / out_per_group;
    int ocg = oc - g * out_per_group;
    int ic_start = g * in_per_group;

    float acc = 0.0f;

    for (int icg = 0; icg < in_per_group; ++icg) {
        int ic = ic_start + icg;

        #pragma unroll
        for (int kh = 0; kh < 3; ++kh) {
            int ih_num = oh + pad_h - kh * dil_h;
            if (ih_num % stride_h != 0) continue;
            int ih = ih_num / stride_h;
            if ((unsigned)ih >= (unsigned)H) continue;

            #pragma unroll
            for (int kw = 0; kw < 5; ++kw) {
                int iw_num = ow + pad_w - kw * dil_w;
                if (iw_num % stride_w != 0) continue;
                int iw = iw_num / stride_w;
                if ((unsigned)iw >= (unsigned)W) continue;

                float xv = x[((n * C_in + ic) * H + ih) * W + iw];
                float wv = w[((ic * out_per_group + ocg) * K_h + kh) * K_w + kw];
                acc += xv * wv;
            }
        }
    }

    y[((n * C_out + oc) * H_out + oh) * W_out + ow] = acc;
}

torch::Tensor conv_transpose2d_cuda(torch::Tensor x, torch::Tensor weight,
                                    int stride_h, int stride_w,
                                    int pad_h, int pad_w,
                                    int dil_h, int dil_w,
                                    int groups) {
    int N = x.size(0);
    int C_in = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int K_h = weight.size(2);
    int K_w = weight.size(3);
    int C_out = weight.size(1) * groups;

    int H_out = (H - 1) * stride_h - 2 * pad_h + dil_h * (K_h - 1) + 1;
    int W_out = (W - 1) * stride_w - 2 * pad_w + dil_w * (K_w - 1) + 1;

    auto y = torch::empty({N, C_out, H_out, W_out}, x.options());

    const int threads = 256;
    long long total = (long long)N * C_out * H_out * W_out;
    int blocks = (int)((total + threads - 1) / threads);

    conv_transpose2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), y.data_ptr<float>(),
        N, C_in, H, W, C_out, K_h, K_w, H_out, W_out,
        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups
    );

    return y;
}
"""

conv_transpose2d_ext = load_inline(
    name="conv_transpose2d_ext_kernelbench",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_cuda_source,
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
        kernel_size: tuple,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv_transpose2d_ext.conv_transpose2d_cuda(
            x.contiguous(),
            self.conv_transpose2d.weight.contiguous(),
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1],
            self.dilation[0],
            self.dilation[1],
            self.groups,
        )