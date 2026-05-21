import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 1D Average Pooling
avg_pool_1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void avg_pool_1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int input_length,
    int kernel_size,
    int stride,
    int padding,
    int output_length
) {
    // Each thread handles one element of the output tensor
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size * in_channels * output_length) {
        return;
    }

    // Decompose index into b, c, t_out
    int t_out = idx % output_length;
    int c = (idx / output_length) % in_channels;
    int b = idx / (output_length * in_channels);

    // Calculate the start and end indices in the input sequence for this output position
    // The formula for output index t_out corresponds to input range:
    // start_input = t_out * stride - padding
    // We need to sum valid elements within [start_input, start_input + kernel_size)
    
    int start_input = t_out * stride - padding;
    int end_input = start_input + kernel_size;

    float sum = 0.0f;
    int count = 0;

    // Iterate over the window in the input space
    for (int i = start_input; i < end_input; ++i) {
        if (i >= 0 && i < input_length) {
            // Access element: [b, c, i]
            sum += input[b * in_channels * input_length + c * input_length + i];
            count++;
        }
    }

    // Average pooling divides by the number of valid elements included in the window
    if (count > 0) {
        output[idx] = sum / static_cast<float>(count);
    } else {
        output[idx] = 0.0f;
    }
}

torch::Tensor avg_pool_1d_cuda(torch::Tensor input, int kernel_size, int stride, int padding) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 3, "Input must be a 3D tensor (batch, channels, length)");

    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_length = input.size(2);

    // Calculate output length using PyTorch's formula for AvgPool1d
    // out_len = floor((in_len + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)
    // Assuming dilation=1
    int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::zeros({batch_size, in_channels, output_length}, input.options());

    const int block_size = 256;
    int total_elements = batch_size * in_channels * output_length;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    avg_pool_1d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        input_length,
        kernel_size,
        stride,
        padding,
        output_length
    );

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in avg_pool_1d: ") + cudaGetErrorString(err));
    }

    return output;
}
"""

avg_pool_1d_cpp_source = (
    "torch::Tensor avg_pool_1d_cuda(torch::Tensor input, int kernel_size, int stride, int padding);"
)

# Compile the inline CUDA code for 1D Average Pooling
avg_pool_1d_module = load_inline(
    name="avg_pool_1d",
    cpp_sources=avg_pool_1d_cpp_source,
    cuda_sources=avg_pool_1d_source,
    functions=["avg_pool_1d_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Optimized model that performs 1D Average Pooling using a custom CUDA operator.
    """
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        """
        Initializes the 1D Average Pooling layer with custom CUDA implementation.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to 1.
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies 1D Average Pooling to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, input_length).

        Returns:
            torch.Tensor: Output tensor with 1D Average Pooling applied, shape (batch_size, in_channels, output_length).
        """
        return avg_pool_1d_module.avg_pool_1d_cuda(x, self.kernel_size, self.stride, self.padding)