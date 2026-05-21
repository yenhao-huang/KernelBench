import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused ConvTranspose3d + Scale + BatchNorm + GlobalAvgPool
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Kernel for fused ConvTranspose3d forward + scale + batch norm + global avg pool
// This is a simplified version that combines all operations
// For a real implementation, we would need to implement the full conv transpose,
// but here we'll create a fused kernel that does the scaling, batch norm, and pooling
// while using PyTorch's conv transpose for the convolution part.

// Fused scale + batch norm + global avg pool kernel
__global__ void fused_scale_bn_pool_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float scale_factor,
    float eps,
    int batch_size,
    int channels,
    int depth,
    int height,
    int width
) {
    // Each block handles one channel's spatial dimensions for one batch element
    // We'll use a 2D grid: (batch, channel) and threads over spatial dimensions
    int b = blockIdx.x;
    int c = blockIdx.y;
    
    if (b >= batch_size || c >= channels) return;
    
    int tid = threadIdx.x;
    int spatial_size = depth * height * width;
    
    // Shared memory for partial sum (for global avg pool)
    __shared__ float shared_sum[256]; // assuming blockDim.x <= 256
    shared_sum[tid] = 0.0f;
    
    float sum = 0.0f;
    float var_val = running_var[c];
    float mean_val = running_mean[c];
    float inv_std = rsqrtf(var_val + eps);
    float gamma_val = gamma[c];
    float beta_val = beta[c];
    
    // Process spatial elements in strides
    for (int idx = tid; idx < spatial_size; idx += blockDim.x) {
        // Compute flattened index
        int d = idx / (height * width);
        int hw = idx % (height * width);
        int h = hw / width;
        int w = hw % width;
        
        int input_idx = ((b * channels + c) * depth + d) * height * width + h * width + w;
        float val = input[input_idx];
        
        // Apply scale
        val = val * scale_factor;
        
        // Apply batch norm (using running stats for inference)
        val = (val - mean_val) * inv_std * gamma_val + beta_val;
        
        // Accumulate for global average pooling
        sum += val;
    }
    
    // Store partial sum in shared memory
    shared_sum[tid] = sum;
    __syncthreads();
    
    // Reduce within block (parallel reduction)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    // Thread 0 writes the final pooled value for this (batch, channel)
    if (tid == 0) {
        float avg = shared_sum[0] / (float)spatial_size;
        int output_idx = b * channels + c;
        output[output_idx] = avg;
    }
}

torch::Tensor fused_scale_bn_pool_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float scale_factor,
    float eps
) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);
    
    // Output shape: (batch_size, channels, 1, 1, 1) but we'll return 2D for simplicity
    auto output = torch::empty({batch_size, channels}, input.options());
    
    dim3 grid(batch_size, channels);
    int block_size = 256;
    dim3 block(block_size);
    
    fused_scale_bn_pool_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        scale_factor,
        eps,
        batch_size,
        channels,
        depth,
        height,
        width
    );
    
    // Reshape to (batch_size, channels, 1, 1, 1) to match AdaptiveAvgPool3d output
    output = output.view({batch_size, channels, 1, 1, 1});
    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_scale_bn_pool_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float scale_factor,
    float eps
);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_scale_bn_pool",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_scale_bn_pool_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a 3D transposed convolution, scales the output,
    applies batch normalization, and then performs global average pooling.
    Uses a custom CUDA kernel to fuse scale, batch norm, and global avg pool.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.eps = eps
        self.fused_ops = fused_ops

    def forward(self, x):
        # Step 1: ConvTranspose3d (still using PyTorch's optimized implementation)
        x = self.conv_transpose(x)
        
        # Step 2: Fused scale + batch norm + global avg pool
        # Get batch norm parameters
        gamma = self.batch_norm.weight
        beta = self.batch_norm.bias
        running_mean = self.batch_norm.running_mean
        running_var = self.batch_norm.running_var
        
        # Call fused CUDA kernel
        x = self.fused_ops.fused_scale_bn_pool_cuda(
            x, gamma, beta, running_mean, running_var,
            self.scale_factor, self.eps
        )
        
        return x