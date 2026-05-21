import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA implementation for Transposed 1D Convolution (ConvTranspose1d)
# This kernel performs the operation: out = input * weight^T + bias
# It handles groups, stride, padding, and output_padding.
# Optimized for FP32.

conv_transpose_1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Helper to calculate grid/block dimensions
__device__ int get_grid_size(int size, int block_size) {
    return (size + block_size - 1) / block_size;
}

__global__ void conv_transpose_1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_length,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups
) {
    // Each thread handles one element of the output tensor
    // Output shape: (batch_size, out_channels, out_length)
    // out_length = (in_length - 1) * stride + kernel_size - 2 * padding + output_padding
    
    int out_length = (in_length - 1) * stride + kernel_size - 2 * padding + output_padding;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_length;

    if (idx >= total_elements) return;

    // Decompose index into b, c_out, l_out
    int l_out = idx % out_length;
    int temp = idx / out_length;
    int c_out = temp % out_channels;
    int b = temp / out_channels;

    // Initialize output to 0 (since we are accumulating)
    float sum = 0.0f;

    // Determine the input channel index corresponding to this output channel
    // For grouped convolutions, c_in is within the group
    int group_idx = c_out / groups;
    int c_in_base = group_idx * (in_channels / groups);
    int c_in_offset = c_out % (in_channels / groups);

    // Iterate over kernel positions and input length
    // The relationship between output position l_out and input position l_in is:
    // l_out = l_in * stride + k - padding
    // => l_in = (l_out - k + padding) / stride
    // We need to find all k such that 0 <= l_in < in_length
    
    for (int k = 0; k < kernel_size; ++k) {
        int l_in = (l_out - k + padding);
        
        // Check if l_in is divisible by stride and within bounds
        if (l_in % stride == 0) {
            l_in /= stride;
            
            if (l_in >= 0 && l_in < in_length) {
                // Access weight: shape (out_channels, in_channels/groups, kernel_size)
                // Weight index for this specific output channel and input channel offset
                int w_idx = c_out * (in_channels / groups) * kernel_size + c_in_offset * kernel_size + k;
                
                // Access input: shape (batch_size, in_channels, in_length)
                int i_idx = b * in_channels * in_length + (c_in_base + c_in_offset) * in_length + l_in;

                sum += weight[w_idx] * input[i_idx];
            }
        }
    }

    // Add bias if present
    if (bias != nullptr) {
        sum += bias[c_out];
    }

    output[idx] = sum;
}

torch::Tensor conv_transpose_1d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int output_padding,
    int groups
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_length = input.size(2);
    
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);

    // Calculate output length
    int out_length = (in_length - 1) * stride + kernel_size - 2 * padding + output_padding;
    
    TORCH_CHECK(out_length > 0, "Output length must be positive");

    auto output = torch::zeros({batch_size, out_channels, out_length}, input.options());

    const int block_size = 256;
    int total_elements = batch_size * out_channels * out_length;
    int num_blocks = get_grid_size(total_elements, block_size);

    const float* bias_ptr = bias.numel() > 0 ? bias.data_ptr<float>() : nullptr;

    conv_transpose_1d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_length,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups
    );

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in conv_transpose_1d: ") + cudaGetErrorString(err));
    }

    return output;
}
"""

conv_transpose_1d_cpp_source = (
    "torch::Tensor conv_transpose_1d_cuda("
    "torch::Tensor input,"
    "torch::Tensor weight,"
    "torch::Tensor bias,"
    "int stride,"
    "int padding,"
    "int output_padding,"
    "int groups"
    ");"
);

// Compile the inline CUDA code
conv_transpose_1d_module = load_inline(
    name="conv_transpose_1d_cuda",
    cpp_sources=conv_transpose_1d_cpp_source,
    cuda_sources=conv_transpose_1d_source,
    functions=["conv_transpose_1d_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Performs a transposed 1D convolution operation using custom CUDA kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Store parameters for forward pass
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Initialize weights and bias using PyTorch's default initialization
        # Weight shape for ConvTranspose1d: (in_channels, out_channels/groups, kernel_size)
        # Note: nn.ConvTranspose1d weight shape is (out_channels, in_channels/groups, kernel_size)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=0.0)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution using custom CUDA kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        return conv_transpose_1d_module.conv_transpose_1d_cuda(
            x,
            self.weight,
            self.bias if self.bias is not None else torch.empty(0),
            self.stride,
            self.padding,
            self.output_padding,
            self.groups
        )

def get_inputs():
    # randomly generate input tensors based on the model architecture
    batch_size = 64
    in_channels = 128
    out_channels = 128
    kernel_size = 3
    length = 65536
    
    x = torch.rand(batch_size, in_channels, length).cuda()
    return [x]

def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    batch_size = 64
    in_channels = 128
    out_channels = 128
    kernel_size = 3
    
    # Return parameters needed to instantiate ModelNew
    return [in_channels, out_channels, kernel_size]