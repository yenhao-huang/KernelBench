import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# C++ declaration
conv_transpose3d_cpp_source = """
torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int groups);
"""

# CUDA implementation
conv_transpose3d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int in_channels, int out_channels,
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out,
    int kD, int kH, int kW,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int groups,
    int out_channels_per_group,
    int in_channels_per_group)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * out_channels * D_out * H_out * W_out;
    if (idx >= total_elements) return;

    // Unflatten output index
    int w_out = idx % W_out;
    int tmp = idx / W_out;
    int h_out = tmp % H_out;
    tmp /= H_out;
    int d_out = tmp % D_out;
    tmp /= D_out;
    int oc = tmp % out_channels;
    int n = tmp / out_channels;

    float val = 0.0f;
    int group = oc / out_channels_per_group;
    int oc_in_group = oc % out_channels_per_group;
    int ic_start = group * in_channels_per_group;
    int ic_end = ic_start + in_channels_per_group;

    int weight_base_stride_ic = out_channels_per_group * kD * kH * kW;
    int weight_stride_oc = kD * kH * kW;

    for (int ic = ic_start; ic < ic_end; ++ic) {
        int weight_ic_offset = ic * weight_base_stride_ic;
        for (int kd = 0; kd < kD; ++kd) {
            int d_in = d_out + padding_d - kd;
            if (d_in % stride_d != 0) continue;
            d_in /= stride_d;
            if (d_in < 0 || d_in >= D_in) continue;

            for (int kh = 0; kh < kH; ++kh) {
                int h_in = h_out + padding_h - kh;
                if (h_in % stride_h != 0) continue;
                h_in /= stride_h;
                if (h_in < 0 || h_in >= H_in) continue;

                for (int kw = 0; kw < kW; ++kw) {
                    int w_in = w_out + padding_w - kw;
                    if (w_in % stride_w != 0) continue;
                    w_in /= stride_w;
                    if (w_in < 0 || w_in >= W_in) continue;

                    float w_val = weight[weight_ic_offset + oc_in_group * weight_stride_oc + 
                                         kd * (kH * kW) + kh * kW + kw];
                    int input_idx = n * (in_channels * D_in * H_in * W_in) +
                                    ic * (D_in * H_in * W_in) +
                                    d_in * (H_in * W_in) +
                                    h_in * W_in +
                                    w_in;
                    val += w_val * input[input_idx];
                }
            }
        }
    }

    if (bias != nullptr) {
        val += bias[oc];
    }
    output[idx] = val;
}

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int groups)
{
    // Ensure contiguous tensors
    input = input.contiguous();
    weight = weight.contiguous();

    int N = input.size(0);
    int in_channels = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);

    int out_channels = weight.size(1) * groups;
    int kD = weight.size(2);
    int kH = weight.size(3);
    int kW = weight.size(4);

    int stride_d = stride[0];
    int stride_h = stride[1];
    int stride_w = stride[2];

    int padding_d = padding[0];
    int padding_h = padding[1];
    int padding_w = padding[2];

    int output_padding_d = output_padding[0];
    int output_padding_h = output_padding[1];
    int output_padding_w = output_padding[2];

    // Output spatial dimensions
    int D_out = (D_in - 1) * stride_d - 2 * padding_d + kD + output_padding_d;
    int H_out = (H_in - 1) * stride_h - 2 * padding_h + kH + output_padding_h;
    int W_out = (W_in - 1) * stride_w - 2 * padding_w + kW + output_padding_w;

    auto output = torch::empty({N, out_channels, D_out, H_out, W_out}, input.options());

    float* bias_ptr = nullptr;
    if (bias.has_value() && bias.value().defined()) {
        bias_ptr = bias.value().contiguous().data_ptr<float>();
    }

    int in_channels_per_group = in_channels / groups;
    int out_channels_per_group = out_channels / groups;

    int total_elements = N * out_channels * D_out * H_out * W_out;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    conv_transpose3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, in_channels, out_channels,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        kD, kH, kW,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        groups,
        out_channels_per_group,
        in_channels_per_group
    );

    return output;
}
"""

# Compile the custom CUDA operator
conv_transpose3d_module = load_inline(
    name="conv_transpose3d_custom",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_cuda_source,
    functions=["conv_transpose3d_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA kernel for transposed 3D convolution.
    Replaces nn.ConvTranspose3d with a fused, low-overhead implementation.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0),
                 output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.use_bias = bias

        # Weight shape: (in_channels, out_channels // groups, kD, kH, kW)
        weight_shape = (in_channels, out_channels // groups, *kernel_size)
        self.weight = nn.Parameter(torch.empty(*weight_shape))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

        # Initialize weights similarly to nn.ConvTranspose3d
        self.reset_parameters()

        # Store the compiled CUDA function
        self.conv_transpose3d_cuda = conv_transpose3d_module.conv_transpose3d_cuda

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution using custom CUDA operator.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D_in, H_in, W_in).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D_out, H_out, W_out).
        """
        bias = self.bias if self.use_bias else None
        return self.conv_transpose3d_cuda(
            x,
            self.weight,
            bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups
        )