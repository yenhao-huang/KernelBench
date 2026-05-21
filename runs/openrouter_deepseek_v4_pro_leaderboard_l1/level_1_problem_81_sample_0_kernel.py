import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# CUDA source for transposed 2D convolution (naive implementation)
conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const bool use_bias,
    const float* __restrict__ bias) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_height * output_width;
    if (idx >= total_elements) return;

    int w_out = idx % output_width;
    int tmp = idx / output_width;
    int h_out = tmp % output_height;
    tmp /= output_height;
    int oc = tmp % out_channels;
    int n = tmp / out_channels;

    float value = 0.0f;

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int h_in = h_out + padding - dilation * kh;
                int w_in = w_out + padding - dilation * kw;

                if (h_in % stride == 0 && w_in % stride == 0) {
                    h_in /= stride;
                    w_in /= stride;
                    if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                        float w_val = weight[((ic * out_channels + oc) * kernel_size + kh) * kernel_size + kw];
                        float x_val = input[((n * in_channels + ic) * input_height + h_in) * input_width + w_in];
                        value += x_val * w_val;
                    }
                }
            }
        }
    }

    if (use_bias) {
        value += bias[oc];
    }

    output[idx] = value;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    const int out_channels = weight.size(1);
    const int kernel_size = weight.size(2);   // square kernel assumed

    // Output spatial dimensions formula for transposed convolution
    int output_height = (input_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    int output_width  = (input_width  - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, input.options());

    const int total_threads = batch_size * out_channels * output_height * output_width;
    const int block_size = 256;
    const int num_blocks = (total_threads + block_size - 1) / block_size;

    const float* bias_ptr = bias.has_value() ? bias->data_ptr<float>() : nullptr;
    bool use_bias = bias.has_value();

    conv_transpose2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_height, input_width,
        output_height, output_width,
        kernel_size,
        stride, padding, dilation,
        use_bias, bias_ptr);

    return output;
}
"""

conv_transpose2d_cpp_source = (
    "torch::Tensor conv_transpose2d_cuda("
    "    torch::Tensor input,"
    "    torch::Tensor weight,"
    "    torch::optional<torch::Tensor> bias,"
    "    int stride,"
    "    int padding,"
    "    int dilation);"
)

# Compile the inline CUDA code for transposed convolution
conv_transpose2d = load_inline(
    name="conv_transpose2d_cuda",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Performs a 2D transposed convolution operation using a custom CUDA kernel.
    Args same as the original Model.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Weight shape: (in_channels, out_channels, kernel_size, kernel_size)
        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.custom_conv = conv_transpose2d

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size * self.kernel_size
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution using the custom CUDA kernel.
        """
        return self.custom_conv.conv_transpose2d_cuda(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
        )