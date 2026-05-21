import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# CUDA source code for transposed 1D convolution
conv_transpose1d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <optional>

__global__ void conv_transpose1d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int B, int C_in, int C_out, int L_in, int L_out,
    int kernel_size, int stride, int padding, int dilation,
    bool use_bias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * C_out * L_out;
    if (idx >= total_elements) return;

    // Decode flattened index into batch, output channel, and output position
    int b = idx / (C_out * L_out);
    int remainder = idx % (C_out * L_out);
    int oc = remainder / L_out;
    int ox = remainder % L_out;

    float sum = 0.0f;
    for (int ic = 0; ic < C_in; ++ic) {
        for (int k = 0; k < kernel_size; ++k) {
            int numerator = ox + padding - k * dilation;
            if (numerator % stride != 0) continue;
            int ix = numerator / stride;
            if (ix >= 0 && ix < L_in) {
                float x_val = x[b * C_in * L_in + ic * L_in + ix];
                float w_val = w[ic * C_out * kernel_size + oc * kernel_size + k];
                sum += x_val * w_val;
            }
        }
    }
    if (use_bias) {
        sum += bias[oc];
    }
    out[idx] = sum;
}

torch::Tensor conv_transpose1d_cuda(
    torch::Tensor x,
    torch::Tensor w,
    std::optional<torch::Tensor> bias_opt,
    int stride, int padding, int dilation)
{
    const auto B = x.size(0);
    const auto C_in = x.size(1);
    const auto L_in = x.size(2);
    const auto C_out = w.size(1);
    const auto kernel_size = w.size(2);

    int L_out = (L_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    if (L_out <= 0) {
        throw std::invalid_argument("Calculated output length is non-positive.");
    }

    auto out = torch::empty({B, C_out, L_out}, x.options());

    bool use_bias = bias_opt.has_value();
    const float* bias_ptr = use_bias ? bias_opt->data_ptr<float>() : nullptr;

    const int threads = 256;
    int total_elements = B * C_out * L_out;
    const int blocks = (total_elements + threads - 1) / threads;

    conv_transpose1d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        bias_ptr,
        out.data_ptr<float>(),
        B, C_in, C_out, L_in, L_out,
        kernel_size, stride, padding, dilation,
        use_bias);

    return out;
}
"""

conv_transpose1d_cpp_source = "torch::Tensor conv_transpose1d_cuda(torch::Tensor x, torch::Tensor w, std::optional<torch::Tensor> bias_opt, int stride, int padding, int dilation);"

# Compile the custom CUDA operator
conv_transpose1d_op = load_inline(
    name="conv_transpose1d_cuda",
    cpp_sources=conv_transpose1d_cpp_source,
    cuda_sources=conv_transpose1d_cuda_source,
    functions=["conv_transpose1d_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Performs a transposed 1D convolution operation with asymmetric input and square kernel.
    Supports padding, striding, and dilation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Weight shape: (in_channels, out_channels, kernel_size)
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Default initialization matching nn.ConvTranspose1d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        return conv_transpose1d_op.conv_transpose1d_cuda(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation
        )