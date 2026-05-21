import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Max Pooling 3D
maxpool3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Helper to get index from input coordinates
__device__ inline int get_input_index(int n, int c, int d, int h, int w, int stride_n, int stride_c, int stride_d, int stride_h, int stride_w, int dim1, int dim2, int dim3) {
    return n * stride_n + c * stride_c + d * stride_d + h * stride_h + w * stride_w;
}

__global__ void maxpool3d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int in_dim1,
    int in_dim2,
    int in_dim3,
    int out_dim1,
    int out_dim2,
    int out_dim3,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    // Each thread handles one output element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int total_out_elements = batch_size * channels * out_dim1 * out_dim2 * out_dim3;
    
    if (idx >= total_out_elements) return;

    // Decode linear index to 5D coordinates (n, c, d, h, w) for output
    int temp = idx;
    int w_out = temp % out_dim3;
    temp /= out_dim3;
    int h_out = temp % out_dim2;
    temp /= out_dim2;
    int d_out = temp % out_dim1;
    temp /= out_dim1;
    int c = temp % channels;
    int n = temp / channels;

    // Calculate the starting position in the input space for this output element
    // Input coordinate = padding + d_out * stride
    int start_d = d_out * stride - padding;
    int start_h = h_out * stride - padding;
    int start_w = w_out * stride - padding;

    // Determine the bounds of the kernel window in input space
    // We need to iterate over the kernel dimensions k_d, k_h, k_w
    // The actual input coordinate is: start_d + k_d * dilation
    
    float max_val = -FLT_MAX;

    // Loop over kernel depth
    for (int k_d = 0; k_d < kernel_size; ++k_d) {
        int d_in = start_d + k_d * dilation;
        if (d_in < 0 || d_in >= in_dim1) continue;

        // Loop over kernel height
        for (int k_h = 0; k_h < kernel_size; ++k_h) {
            int h_in = start_h + k_h * dilation;
            if (h_in < 0 || h_in >= in_dim2) continue;

            // Loop over kernel width
            for (int k_w = 0; k_w < kernel_size; ++k_w) {
                int w_in = start_w + k_w * dilation;
                if (w_in < 0 || w_in >= in_dim3) continue;

                // Get input value
                // Strides for input tensor: N, C, D, H, W
                // Assuming contiguous memory layout: [N][C][D][H][W]
                // stride_n = C * D * H * W
                // stride_c = D * H * W
                // stride_d = H * W
                // stride_h = W
                // stride_w = 1
                
                int input_idx = n * (channels * in_dim1 * in_dim2 * in_dim3) + 
                                c * (in_dim1 * in_dim2 * in_dim3) + 
                                d_in * (in_dim2 * in_dim3) + 
                                h_in * in_dim3 + 
                                w_in;
                
                float val = input[input_idx];
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
    }

    // Write output value
    int output_idx = idx;
    output[output_idx] = max_val;
}

torch::Tensor maxpool3d_cuda(torch::Tensor input, int kernel_size, int stride, int padding, int dilation) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 5, "Input must be 5D (N, C, D, H, W)");

    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto in_dim1 = input.size(2);
    auto in_dim2 = input.size(3);
    auto in_dim3 = input.size(4);

    // Calculate output dimensions
    // Formula: floor((input_size + 2*padding - dilation*(kernel_size-1) - 1) / stride) + 1
    int out_dim1 = (in_dim1 + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_dim2 = (in_dim2 + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_dim3 = (in_dim3 + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    // Handle ceil_mode or invalid dimensions gracefully if needed, 
    // but standard floor formula is used here as per typical maxpool behavior without ceil_mode.
    // If out_dim <= 0, return empty tensor or handle error.
    if (out_dim1 <= 0 || out_dim2 <= 0 || out_dim3 <= 0) {
        return torch::empty({batch_size, channels, 0, 0, 0}, input.options());
    }

    auto output = torch::empty({batch_size, channels, out_dim1, out_dim2, out_dim3}, input.options());

    const int block_size = 256;
    int total_out_elements = batch_size * channels * out_dim1 * out_dim2 * out_dim3;
    int num_blocks = (total_out_elements + block_size - 1) / block_size;

    maxpool3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        in_dim1,
        in_dim2,
        in_dim3,
        out_dim1,
        out_dim2,
        out_dim3,
        kernel_size,
        stride,
        padding,
        dilation
    );

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\\n", cudaGetErrorString(err));
    }

    return output;
}
"""

maxpool3d_cpp_source = (
    "torch::Tensor maxpool3d_cuda(torch::Tensor input, int kernel_size, int stride, int padding, int dilation);"
)

# Compile the inline CUDA code for Max Pooling 3D
maxpool3d_module = load_inline(
    name="maxpool3d_cuda",
    cpp_sources=maxpool3d_cpp_source,
    cuda_sources=maxpool3d_source,
    functions=["maxpool3d_cuda"],
    verbose=False,
    extra_cflags=["-O2"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Optimized model that performs Max Pooling 3D using a custom CUDA operator.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        """
        Initializes the Max Pooling 3D layer.

        Args:
            kernel_size (int): Size of the kernel for the max pooling operation.
            stride (int, optional): Stride of the pooling operation. Defaults to None, which means stride is equal to kernel_size.
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            return_indices (bool, optional): Whether to return indices of the maximum values. Defaults to False.
            ceil_mode (bool, optional): When True, the output size is ceil(input_size / stride) instead of floor. Defaults to False.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 3D to the input tensor using custom CUDA operator.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, dim1, dim2, dim3).

        Returns:
            torch.Tensor: Output tensor with Max Pooling 3D applied.
        """
        return maxpool3d_module.maxpool3d_cuda(x, self.kernel_size, self.stride, self.padding, self.dilation)