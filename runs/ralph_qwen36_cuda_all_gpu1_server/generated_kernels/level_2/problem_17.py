import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Instance Normalization followed by division.
# We fuse the InstanceNorm (mean/variance calculation, normalization) and the final division
# into a single kernel to reduce memory bandwidth pressure and improve performance.
instance_norm_divide_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void instance_norm_divide_kernel(
    const float* input, 
    float* output, 
    int batch_size, 
    int channels, 
    int spatial_dim,
    float eps,
    float divide_by) {
    
    // Each thread block handles one channel of one sample in the batch.
    // However, to maximize occupancy and handle variable spatial dimensions efficiently,
    // we often launch one thread per element or use a grid-stride loop.
    // Here, we assume a standard approach where threads map to elements.
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * spatial_dim;
    
    if (idx < total_elements) {
        // Determine which sample and channel this element belongs to
        int spatial_idx = idx % spatial_dim;
        int temp = idx / spatial_dim;
        int c = temp % channels;
        int b = temp / channels;
        
        // Calculate the global index for the start of this (b, c) slice
        int base_idx = (b * channels + c) * spatial_dim;
        
        // Step 1: Compute Mean
        float sum = 0.0f;
        for (int i = 0; i < spatial_dim; ++i) {
            sum += input[base_idx + i];
        }
        float mean = sum / spatial_dim;
        
        // Step 2: Compute Variance
        float sq_sum = 0.0f;
        for (int i = 0; i < spatial_dim; ++i) {
            float diff = input[base_idx + i] - mean;
            sq_sum += diff * diff;
        }
        float var = sq_sum / spatial_dim;
        
        // Step 3: Normalize and Divide
        float inv_std = rsqrtf(var + eps);
        output[idx] = (input[idx] - mean) * inv_std / divide_by;
    }
}

torch::Tensor instance_norm_divide_cuda(torch::Tensor input, float eps, float divide_by) {
    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto output = torch::empty_like(input);
    
    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int spatial_dim = height * width;
    int total_elements = batch_size * channels * spatial_dim;
    
    if (total_elements == 0) {
        return output;
    }
    
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    instance_norm_divide_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        batch_size, 
        channels, 
        spatial_dim,
        eps,
        divide_by
    );
    
    return output;
}
"""

instance_norm_divide_cpp_source = (
    "torch::Tensor instance_norm_divide_cuda(torch::Tensor input, float eps, float divide_by);"
)

# Compile the inline CUDA code
instance_norm_divide = load_inline(
    name="instance_norm_divide",
    cpp_sources=instance_norm_divide_cpp_source,
    cuda_sources=instance_norm_divide_source,
    functions=["instance_norm_divide_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA operator for Instance Normalization and Division.
    The convolution is left as PyTorch's optimized cuDNN implementation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # We don't need the PyTorch InstanceNorm anymore as it's fused into the custom op
        self.divide_by = divide_by
        self.eps = 1e-5

    def forward(self, x):
        x = self.conv(x)
        # Use custom fused kernel for InstanceNorm + Division
        x = instance_norm_divide.instance_norm_divide_cuda(x, self.eps, self.divide_by)
        return x