import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source code for depthwise asymmetric convolution
depthwise_conv2d_asym_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void depthwise_conv2d_asymmetric_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int height,
    int width,
    int kernel_h,
    int stride,
    int padding,
    int dilation,
    int out_h,
    int out_w,
    bool use_bias)
{
    int total = batch_size * channels * out_h * out_w;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    int b = tid / (channels * out_h * out_w);
    int rem = tid % (channels * out_h * out_w);
    int c = rem / (out_h * out_w);
    rem = rem % (out_h * out_w);
    int h_out = rem / out_w;
    int w_out = rem % out_w;

    float val = 0.0f;
    for (int k = 0; k < kernel_h; ++k) {
        int h_in = h_out * stride + k * dilation - padding;
        int w_in = w_out * stride - padding;  // kernel width is 1, only position 0
        if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
            int input_idx = ((b * channels + c) * height + h_in) * width + w_in;
            int weight_idx = c * kernel_h + k; // weight shape: [C, 1, K, 1] -> (c,0,k,0) offset = c*K + k
            val += input[input_idx] * weight[weight_idx];
        }
    }
    if (use_bias) {
        val += bias[c];
    }
    output[tid] = val;
}

torch::Tensor depthwise_conv2d_asymmetric_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation)
{
    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int kernel_h = weight.size(2);  // weight shape: [C, 1, K, 1]

    int out_h = (height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_w = (width + 2 * padding - dilation * (1 - 1) - 1) / stride + 1;

    auto output = torch::empty({batch, channels, out_h, out_w}, input.options());

    int total = batch * channels * out_h * out_w;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    bool use_bias = bias.has_value();
    const float* bias_ptr = use_bias ? bias.value().data_ptr<float>() : nullptr;

    depthwise_conv2d_asymmetric_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch, channels, height, width,
        kernel_h, stride, padding, dilation,
        out_h, out_w, use_bias);

    return output;
}
"""

depthwise_conv2d_asym_cpp_source = (
    "torch::Tensor depthwise_conv2d_asymmetric_cuda("
    "torch::Tensor input, torch::Tensor weight, "
    "torch::optional<torch::Tensor> bias, "
    "int stride, int padding, int dilation);"
)

# Compile the inline CUDA code
depthwise_conv2d_asym = load_inline(
    name="depthwise_conv2d_asym",
    cpp_sources=depthwise_conv2d_asym_cpp_source,
    cuda_sources=depthwise_conv2d_asym_source,
    functions=["depthwise_conv2d_asymmetric_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution with a square input and an asymmetric kernel
    using a custom CUDA operator for speedup.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Weight shape: (in_channels, 1, kernel_size, 1)
        weight = torch.empty(in_channels, 1, kernel_size, 1)
        self.weight = nn.Parameter(weight)
        if bias:
            self.bias = nn.Parameter(torch.empty(in_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.custom_op = depthwise_conv2d_asym

    def reset_parameters(self):
        # Mimics nn.Conv2d initialization
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.custom_op.depthwise_conv2d_asymmetric_cuda(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation
        )


# Test code (unchanged from original)
batch_size = 64
in_channels = 8
kernel_size = 3
width = 512
height = 512
stride = 1
padding = 0
dilation = 1

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, kernel_size, stride, padding, dilation]