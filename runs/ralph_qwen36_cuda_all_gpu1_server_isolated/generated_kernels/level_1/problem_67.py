import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 1D Convolution with bias
# This implementation uses a naive but correct approach. For production, one would use cuDNN or cutlass,
# but here we implement a custom kernel to demonstrate inline CUDA integration.
# We will fuse the convolution and bias addition if bias is present.

conv1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Kernel for 1D Convolution without bias
__global__ void conv1d_no_bias_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int length_in,
    int kernel_size,
    int stride,
    int padding,
    int dilation) 
{
    // Each thread computes one element of the output tensor: (n, c_out, l_out)
    int n = blockIdx.z;
    int c_out = blockIdx.y;
    int l_out = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= batch_size || c_out >= out_channels || l_out >= length_in) return;

    // Calculate the start index in the input sequence for this output position
    int l_start = l_out * stride - padding;
    
    float sum = 0.0f;
    
    // Iterate over input channels and kernel positions
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int k = 0; k < kernel_size; ++k) {
            int l_in = l_start + k * dilation;
            
            // Check bounds for input sequence length
            if (l_in >= 0 && l_in < length_in) {
                // Input index: (n, c_in, l_in)
                int input_idx = n * (in_channels * length_in) + c_in * length_in + l_in;
                
                // Weight index: (c_out, c_in, k)
                int weight_idx = c_out * (in_channels * kernel_size) + c_in * kernel_size + k;
                
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    
    // Output index: (n, c_out, l_out)
    int output_idx = n * (out_channels * length_in) + c_out * length_in + l_out;
    output[output_idx] = sum;
}

// Kernel for 1D Convolution with bias
__global__ void conv1d_with_bias_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int length_in,
    int kernel_size,
    int stride,
    int padding,
    int dilation) 
{
    int n = blockIdx.z;
    int c_out = blockIdx.y;
    int l_out = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= batch_size || c_out >= out_channels || l_out >= length_in) return;

    int l_start = l_out * stride - padding;
    
    float sum = 0.0f;
    
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int k = 0; k < kernel_size; ++k) {
            int l_in = l_start + k * dilation;
            
            if (l_in >= 0 && l_in < length_in) {
                int input_idx = n * (in_channels * length_in) + c_in * length_in + l_in;
                int weight_idx = c_out * (in_channels * kernel_size) + c_in * kernel_size + k;
                
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    
    int output_idx = n * (out_channels * length_in) + c_out * length_in + l_out;
    output[output_idx] = sum + bias[c_out];
}

torch::Tensor conv1d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation) 
{
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto length_in = input.size(2);
    
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);
    
    // Calculate output length
    int length_out = (length_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, length_out}, input.options());
    
    const int block_size = 256;
    const int num_blocks_x = (length_out + block_size - 1) / block_size;
    dim3 grid(num_blocks_x, out_channels, batch_size);
    dim3 block(block_size);
    
    if (bias.numel() > 0) {
        conv1d_with_bias_kernel<<<grid, block>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            in_channels,
            out_channels,
            length_in,
            kernel_size,
            stride,
            padding,
            dilation
        );
    } else {
        conv1d_no_bias_kernel<<<grid, block>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            in_channels,
            out_channels,
            length_in,
            kernel_size,
            stride,
            padding,
            dilation
        );
    }
    
    return output;
}
"""

conv1d_cpp_source = (
    "torch::Tensor conv1d_cuda("
    "torch::Tensor input,"
    "torch::Tensor weight,"
    "torch::Tensor bias,"
    "int stride,"
    "int padding,"
    "int dilation"
    ");"
)

# Compile the inline CUDA code
conv1d_module = load_inline(
    name="conv1d_cuda_module",
    cpp_sources=conv1d_cpp_source,
    cuda_sources=conv1d_source,
    functions=["conv1d_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"]
)


class ModelNew(nn.Module):
    """
    Performs a standard 1D convolution operation using custom CUDA operators.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # We must store the parameters to pass them to the custom CUDA kernel
        # Note: In a real scenario, you might want to register these as buffers or handle them differently
        # to ensure they are moved to GPU correctly. Here we use buffers for simplicity and correctness.
        self.register_buffer('weight', torch.zeros(out_channels, in_channels // groups, kernel_size))
        if bias:
            self.register_buffer('bias', torch.zeros(out_channels))
        else:
            self.register_buffer('bias', torch.zeros(0)) # Empty tensor if no bias
            
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights with random values similar to nn.Conv1d default initialization
        nn.init.kaiming_uniform_(self.weight, a=0.0)
        if bias.numel() > 0:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 1D convolution using custom CUDA kernel.
        """
        # Ensure inputs are on the same device as the model parameters
        if x.device != self.weight.device:
            x = x.to(self.weight.device)
            
        return conv1d_module.conv1d_cuda(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation
        )

import math