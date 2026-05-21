import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for a fused Conv2d + Bias operation.
# This implementation uses a tiled approach to handle the convolution.
# For simplicity and robustness in a single-file format, we implement a 
# direct convolution kernel that handles dilation, padding, and stride.
conv2d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv2d_bias_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int in_h, int in_w,
    int out_channels, int kernel_h, int kernel_w,
    int out_h, int out_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_h * out_w;

    if (idx < total_elements) {
        // Decompose index
        int w_out = idx % out_w;
        int h_out = (idx / out_w) % out_h;
        int c_out = (idx / (out_w * out_h)) % out_channels;
        int b = idx / (out_w * out_h * out_channels);

        float sum = (bias != nullptr) ? bias[c_out] : 0.0f;

        for (int c_in = 0; c_in < in_channels; ++c_in) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    // Calculate input coordinates with dilation and padding
                    int h_in = h_out * stride_h - pad_h + kh * dilation_h;
                    int w_in = w_out * stride_w - pad_w + kw * dilation_w;

                    if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
                        int input_idx = (((b * in_channels + c_in) * in_h + h_in) * in_w) + w_in;
                        int weight_idx = (((c_out * in_channels + c_in) * kernel_h + kh) * kernel_w) + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        output[idx] = sum;
    }
}

torch::Tensor conv2d_bias_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    at::optional<torch::Tensor> bias,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w) 
{
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_h = input.size(2);
    const int in_w = input.size(3);

    const int out_channels = weight.size(0);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);

    const int out_h = (in_h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int out_w = (in_w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, input.options());

    int total_elements = batch_size * out_channels * out_h * out_w;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;

    conv2d_bias_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size, in_channels, in_h, in_w,
        out_channels, kernel_h, kernel_w,
        out_h, out_w,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w
    );

    return output;
}
"""

conv2d_cpp_source = """
torch::Tensor conv2d_bias_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    at::optional<torch::Tensor> bias,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w);
"""

# Compile the inline CUDA code
conv_cuda_module = load_inline(
    name="conv_cuda_module",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_cuda_source,
    functions=["conv2d_bias_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.conv_cuda = conv_cuda_module.conv2d_bias_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are contiguous for the CUDA kernel
        x = x.contiguous()
        weight = self.weight.contiguous()
        bias = self.bias.contiguous() if self.bias is not None else None
        
        return self.conv_cuda(
            x, 
            weight, 
            bias, 
            self.stride[0], self.stride[1], 
            self.padding[0], self.padding[1], 
            self.dilation[0], self.dilation[1]
        )