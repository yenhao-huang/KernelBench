import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA source for 3D convolution
conv3d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv3d_direct_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int B,
    int C_in,
    int C_out,
    int D,
    int H,
    int W,
    int KD,
    int KH,
    int KW,
    int stride_d,
    int stride_h,
    int stride_w,
    int pad_d,
    int pad_h,
    int pad_w,
    int dilation_d,
    int dilation_h,
    int dilation_w,
    int groups,
    int out_D,
    int out_H,
    int out_W,
    int C_in_per_group,
    int C_out_per_group
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = B * C_out * out_D * out_H * out_W;
    if (idx >= total_out) return;

    // Unmap linear index to output coordinates
    int w_out = idx % out_W;
    int idx_ = idx / out_W;
    int h_out = idx_ % out_H;
    idx_ /= out_H;
    int d_out = idx_ % out_D;
    idx_ /= out_D;
    int c_out = idx_ % C_out;
    idx_ /= C_out;
    int b = idx_;

    int group = c_out / C_out_per_group;
    int c_out_local = c_out % C_out_per_group;

    float sum = 0.0f;
    for (int c_in = 0; c_in < C_in_per_group; ++c_in) {
        int ch_in = group * C_in_per_group + c_in;
        for (int kd = 0; kd < KD; ++kd) {
            int in_d = d_out * stride_d - pad_d + kd * dilation_d;
            if (in_d < 0 || in_d >= D) continue;
            for (int kh = 0; kh < KH; ++kh) {
                int in_h = h_out * stride_h - pad_h + kh * dilation_h;
                if (in_h < 0 || in_h >= H) continue;
                for (int kw = 0; kw < KW; ++kw) {
                    int in_w = w_out * stride_w - pad_w + kw * dilation_w;
                    if (in_w < 0 || in_w >= W) continue;
                    // Input offset: [b, ch_in, in_d, in_h, in_w]
                    int inp_idx = b * C_in * D * H * W +
                                  ch_in * D * H * W +
                                  in_d * H * W +
                                  in_h * W +
                                  in_w;
                    // Weight offset: [c_out, c_in, kd, kh, kw] (C_out, C_in_per_group, KD, KH, KW)
                    int wgt_idx = c_out * C_in_per_group * KD * KH * KW +
                                  c_in * KD * KH * KW +
                                  kd * KH * KW +
                                  kh * KW +
                                  kw;
                    sum += input[inp_idx] * weight[wgt_idx];
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[c_out];
    }
    output[idx] = sum;
}

torch::Tensor conv3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups
) {
    // Extract dimensions
    int B = input.size(0);
    int C_in = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    int C_out = weight.size(0);
    int C_in_per_group = weight.size(1);
    int KD = weight.size(2);
    int KH = weight.size(3);
    int KW = weight.size(4);

    // Compute output spatial dimensions
    int out_D = (D + 2 * pad_d - dilation_d * (KD - 1) - 1) / stride_d + 1;
    int out_H = (H + 2 * pad_h - dilation_h * (KH - 1) - 1) / stride_h + 1;
    int out_W = (W + 2 * pad_w - dilation_w * (KW - 1) - 1) / stride_w + 1;

    int C_out_per_group = C_out / groups;

    // Allocate output tensor
    auto output = torch::zeros({B, C_out, out_D, out_H, out_W}, input.options());

    const float* bias_ptr = (bias.numel() > 0) ? bias.data_ptr<float>() : nullptr;

    int total_threads = B * C_out * out_D * out_H * out_W;
    const int threads = 256;
    const int blocks = (total_threads + threads - 1) / threads;

    conv3d_direct_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        B, C_in, C_out, D, H, W,
        KD, KH, KW,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dilation_d, dilation_h, dilation_w,
        groups,
        out_D, out_H, out_W,
        C_in_per_group, C_out_per_group
    );

    return output;
}
"""

conv3d_cpp_source = (
    "torch::Tensor conv3d_cuda("
    "    torch::Tensor input,"
    "    torch::Tensor weight,"
    "    torch::Tensor bias,"
    "    int stride_d, int stride_h, int stride_w,"
    "    int pad_d, int pad_h, int pad_w,"
    "    int dilation_d, int dilation_h, int dilation_w,"
    "    int groups"
    ");"
)

# Compile the inline CUDA code for 3D convolution
conv3d_custom = load_inline(
    name="conv3d_custom",
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_cuda_source,
    functions=["conv3d_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized 3D convolution using custom CUDA kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (D, H, W).
        stride (tuple): Stride (D, H, W). Default (1,1,1).
        padding (tuple): Padding (D, H, W). Default (0,0,0).
        dilation (tuple): Dilation (D, H, W). Default (1,1,1).
        groups (int): Number of groups. Default 1.
        bias (bool): If True, adds a learnable bias. Default False.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1),
                 groups=1, bias=False):
        super().__init__()
        # Store the standard convolution to keep parameters (weight, bias)
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias
        )
        self.custom_conv = conv3d_custom
        # Keep convolution parameters for forwarding
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous for the custom kernel
        x = x.contiguous()
        weight = self.conv.weight
        bias = self.conv.bias if self.conv.bias is not None else torch.empty(0, device=x.device)

        return self.custom_conv.conv3d_cuda(
            x,
            weight,
            bias,
            self.stride[0], self.stride[1], self.stride[2],
            self.padding[0], self.padding[1], self.padding[2],
            self.dilation[0], self.dilation[1], self.dilation[2],
            self.groups
        )