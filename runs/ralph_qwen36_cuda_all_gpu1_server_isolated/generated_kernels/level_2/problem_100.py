import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused ConvTranspose3d + Clamp + Divide
# This kernel performs: out = clamp(conv_transpose(x), min) / divisor
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Helper to get grid/block dimensions
dim3 get_grid_dim(int n, int block_size) {
    return dim3((n + block_size - 1) / block_size);
}

__global__ void conv_transpose_3d_kernel(
    const float* input, 
    const float* weight, 
    const float* bias, // Can be null if no bias
    float* output, 
    int batch_size,
    int in_channels,
    int in_depth,
    int in_height,
    int in_width,
    int out_channels,
    int out_depth,
    int out_height,
    int out_width,
    int kernel_depth,
    int kernel_height,
    int kernel_width,
    int stride_d,
    int stride_h,
    int stride_w,
    int pad_d,
    int pad_h,
    int pad_w,
    int dilation_d,
    int dilation_h,
    int dilation_w,
    float min_val,
    float div_val
) {
    // Each thread computes one output element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    if (idx >= total_elements) return;

    // Decompose linear index into coordinates
    int temp = idx;
    int w_idx = temp % out_width;
    temp /= out_width;
    int h_idx = temp % out_height;
    temp /= out_height;
    int d_idx = temp % out_depth;
    temp /= out_depth;
    int c_idx = temp % out_channels;
    int b_idx = temp / out_channels;

    float sum = 0.0f;

    // Iterate over input channels and kernel dimensions
    for (int k_d = 0; k_d < kernel_depth; ++k_d) {
        for (int k_h = 0; k_h < kernel_height; ++k_h) {
            for (int k_w = 0; k_w < kernel_width; ++k_w) {
                // Calculate corresponding input coordinates
                int in_d = d_idx * stride_d - pad_d + k_d * dilation_d;
                int in_h = h_idx * stride_h - pad_h + k_h * dilation_h;
                int in_w = w_idx * stride_w - pad_w + k_w * dilation_w;

                // Check bounds for input tensor
                if (in_d >= 0 && in_d < in_depth &&
                    in_h >= 0 && in_h < in_height &&
                    in_w >= 0 && in_w < in_width) {
                    
                    // Weight index: [out_channels, in_channels, k_d, k_h, k_w]
                    int w_idx_local = c_idx * (in_channels * kernel_depth * kernel_height * kernel_width) + 
                                      0 * (kernel_depth * kernel_height * kernel_width) + // Assuming single input channel group or standard layout? 
                                      // Standard ConvTranspose3d weight shape: [out_channels, in_channels/groups, kD, kH, kW]
                                      // Here we assume groups=1 for simplicity as per nn.ConvTranspose3d default.
                                      // Actually, let's use the full stride logic.
                                      c_idx * (in_channels * kernel_depth * kernel_height * kernel_width) + 
                                      0 * (kernel_depth * kernel_height * kernel_width); // This is wrong if in_channels > 1.
                    
                    // Correct weight indexing:
                    // Weight shape: [out_channels, in_channels, kD, kH, kW]
                    int w_offset = c_idx * (in_channels * kernel_depth * kernel_height * kernel_width) + 
                                   0 * (kernel_depth * kernel_height * kernel_width); // This assumes we are iterating in_channels inside? No.
                    
                    // Let's rewrite the loop to be correct for general in_channels
                }
            }
        }
    }
    
    // The above manual implementation is complex and error-prone for general cases.
    // A better approach for a "custom operator" that replaces PyTorch ops is to use 
    // cuDNN via torch.ops or just implement the math correctly.
    // However, writing a full generic ConvTranspose3d in raw CUDA is very verbose.
    
    // Alternative Strategy: Since we want speedups and freedom, let's focus on the 
    // post-processing (Clamp + Divide) which is trivial to fuse with any output, 
    // OR implement a simplified version if the kernel size is small.
    
    // Given the constraints of "real code" and "compiles", implementing a full generic 
    // ConvTranspose3d from scratch in inline CUDA is extremely risky for correctness 
    // across all padding/stride/dilation combinations without a library like CUTLASS.
    
    // However, we CAN optimize the post-processing steps (Clamp + Divide) into a single kernel 
    // that runs on the output of the standard PyTorch ConvTranspose3d. This is a valid 
    // optimization strategy: replace multiple sequential ops with one fused op.
    
    // Let's define a kernel for Clamp and Divide.
}

__global__ void clamp_divide_kernel(
    const float* input, 
    float* output, 
    int size, 
    float min_val, 
    float div_val
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        if (val < min_val) {
            val = min_val;
        }
        output[idx] = val / div_val;
    }
}

torch::Tensor clamp_divide_cuda(torch::Tensor input, float min_val, float div_val) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    if (size == 0) return output;
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    clamp_divide_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size, min_val, div_val);
    
    return output;
}

"""

clamp_divide_cpp_source = (
    "torch::Tensor clamp_divide_cuda(torch::Tensor input, float min_val, float div_val);"
)

# Compile the inline CUDA code
clamp_divide_ops = load_inline(
    name="clamp_divide_ops",
    cpp_sources=clamp_divide_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["clamp_divide_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    A model that performs a transposed 3D convolution, then fuses the clamp and divide operations 
    into a single custom CUDA kernel for speedup.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.min_value = min_value
        self.divisor = divisor

    def forward(self, x):
        # Perform the convolution using standard PyTorch op (which is highly optimized via cuDNN)
        x = self.conv_transpose(x)
        
        # Replace torch.clamp and division with a single fused CUDA kernel
        x = clamp_divide_ops.clamp_divide_cuda(x, self.min_value, self.divisor)
        
        return x

# Re-defining get_inputs and get_init_inputs to match the required interface for testing/context
def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, min_value, divisor]