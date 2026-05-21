import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for depthwise convolution
depthwise_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void depthwise_conv_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int in_height,
    const int in_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int out_height,
    const int out_width
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = batch_size * channels * out_height * out_width;
    if (tid >= total_out) return;

    int remainder = tid;
    int out_x = remainder % out_width;
    remainder /= out_width;
    int out_y = remainder % out_height;
    remainder /= out_height;
    int channel = remainder % channels;
    remainder /= channels;
    int batch = remainder;

    const float* in_batch_ch = input + (batch * channels + channel) * in_height * in_width;
    const float* filter = weight + channel * kernel_size * kernel_size;

    float sum = 0.0f;
    for (int ky = 0; ky < kernel_size; ++ky) {
        int in_y = out_y * stride + ky - padding;
        if (in_y < 0 || in_y >= in_height) continue;
        for (int kx = 0; kx < kernel_size; ++kx) {
            int in_x = out_x * stride + kx - padding;
            if (in_x < 0 || in_x >= in_width) continue;
            float val = in_batch_ch[in_y * in_width + in_x];
            sum += val * filter[ky * kernel_size + kx];
        }
    }

    if (bias != nullptr) {
        sum += bias[channel];
    }

    float* out_batch_ch = output + (batch * channels + channel) * out_height * out_width;
    out_batch_ch[out_y * out_width + out_x] = sum;
}

torch::Tensor depthwise_conv_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding
) {
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto in_height = input.size(2);
    const auto in_width = input.size(3);
    const auto kernel_size = weight.size(2);
    const auto out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    const auto out_width = (in_width + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, channels, out_height, out_width}, input.options());

    const int total_out = batch_size * channels * out_height * out_width;
    const int threads = 256;
    const int blocks = (total_out + threads - 1) / threads;

    const float* bias_ptr = nullptr;
    if (bias.defined() && bias.numel() > 0) {
        bias_ptr = bias.data_ptr<float>();
    }

    depthwise_conv_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        channels,
        in_height,
        in_width,
        kernel_size,
        stride,
        padding,
        out_height,
        out_width
    );

    return output;
}
"""

depthwise_conv_cpp_source = "torch::Tensor depthwise_conv_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride, int padding);"

# Compile the inline CUDA code for depthwise convolution
depthwise_conv_op = load_inline(
    name="depthwise_conv_op",
    cpp_sources=depthwise_conv_cpp_source,
    cuda_sources=depthwise_conv_source,
    functions=["depthwise_conv_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized depthwise 2D convolution using custom CUDA kernel.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        # Keep nn.Conv2d only as a parameter container; forward is replaced by custom op
        self.conv2d = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels, bias=bias)
        self._depthwise_conv_op = depthwise_conv_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract parameters from the conv layer and call the CUDA kernel
        bias = self.conv2d.bias if self.conv2d.bias is not None else torch.empty(0, device=x.device)
        return self._depthwise_conv_op.depthwise_conv_cuda(
            x.contiguous(),
            self.conv2d.weight,
            bias,
            self.conv2d.stride[0],
            self.conv2d.padding[0]
        )