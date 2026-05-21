import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# The optimization strategy:
# 1. ConvTranspose3d is a heavy operation; we keep it as the standard PyTorch implementation.
# 2. BatchNorm3d (in eval mode) and the spatial mean subtraction can be fused.
#    BatchNorm3d (eval): y = x * scale + shift
#    Mean Subtraction: z = y - mean(y, dim=(2,3,4))
#    Combined: z = (x * scale + shift) - mean(x * scale + shift, dim=(2,3,4))
#    Since mean is a linear operator: mean(x * scale + shift) = scale * mean(x) + shift
#    Therefore: z = x * scale + shift - (scale * mean(x) + shift)
#    z = scale * (x - mean(x))
# This fusion reduces memory bandwidth by performing the scale and subtraction in one pass.

fused_bn_mean_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_bn_mean_kernel(
    const float* __restrict__ input,
    const float* __restrict__ scale,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int spatial_vol) {
    
    // Each block handles one (batch, channel) pair
    int b_idx = blockIdx.y;
    int c_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (b_idx >= batch_size || c_idx >= channels) return;

    int base_idx = (b_idx * channels + c_idx) * spatial_vol;

    // 1. Calculate spatial mean for this (batch, channel)
    float sum = 0.0f;
    for (int i = tid; i < spatial_vol; i += blockDim.x) {
        sum += input[base_idx + i];
    }

    // Reduction in shared memory
    __shared__ float shared_sum[256]; 
    // Note: For simplicity in this inline example, we use a simple block reduction.
    // In a production kernel, we'd use a more robust reduction.
    // However, for the sake of a functional single-kernel implementation:
    
    // We'll use a simple approach: sum all elements in the block via atomicAdd or multiple passes.
    // To keep the code clean and robust for the user, we'll use a two-pass approach or 
    // a single-block reduction if spatial_vol is small, but here we'll use a standard reduction.
    
    // For the sake of this specific task, we'll implement a simple reduction.
    // Since spatial_vol can be large, we use a block-level reduction.
    
    // Re-calculating sum using a more robust method for the kernel:
    // We'll use a simple loop for the mean calculation within the block.
    // To handle large spatial_vol, we'll use atomicAdd to a shared variable.
    __shared__ float block_sum;
    if (tid == 0) block_sum = 0.0f;
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tid; i < spatial_vol; i += blockDim.x) {
        local_sum += input[base_idx + i];
    }
    atomicAdd(&block_sum, local_sum);
    __syncthreads();

    float mean = block_sum / spatial_vol;
    float s = scale[c_idx];

    // 2. Apply: output = scale * (input - mean)
    for (int i = tid; i < spatial_vol; i += blockDim.x) {
        output[base_idx + i] = s * (input[base_idx + i] - mean);
    }
}

torch::Tensor fused_bn_mean_cuda(torch::Tensor input, torch::Tensor scale) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto spatial_vol = input.size(2) * input.size(3) * input.size(4);
    auto output = torch::empty_like(input);

    const int block_size = 256;
    // Grid: x = channels, y = batch_size
    dim3 grid(channels, batch_size);
    dim3 block(block_size);

    fused_bn_mean_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        scale.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        spatial_vol
    );

    return output;
}
"""

fused_bn_mean_cpp_source = "torch::Tensor fused_bn_mean_cuda(torch::Tensor input, torch::Tensor scale);"

fused_op = load_inline(
    name="fused_bn_mean",
    cpp_sources=fused_bn_mean_cpp_source,
    cuda_sources=fused_bn_mean_source,
    functions=["fused_bn_mean_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.fused_op = fused_op

    def forward(self, x):
        # 1. ConvTranspose3d
        x = self.conv_transpose(x)
        
        # 2. Prepare BatchNorm parameters for fusion
        # In eval mode, BatchNorm is: y = x * scale + shift
        # where scale = gamma / sqrt(var + eps)
        # and shift = beta - running_mean * scale
        # However, the original model performs: x = x - mean(x) AFTER BatchNorm.
        # As derived: x_final = scale * (x_input - mean(x_input))
        # This is because the 'shift' term cancels out during mean subtraction.
        
        if self.training:
            # If training, we fall back to standard PyTorch to handle momentum/running stats
            x = self.batch_norm(x)
            x = x - torch.mean(x, dim=(2, 3, 4), keepdim=True)
        else:
            # In eval mode, we use the fused kernel
            # We need the 'scale' parameter: gamma / sqrt(var + eps)
            # Note: BatchNorm3d.weight is gamma, BatchNorm3d.running_var is var
            eps = self.batch_norm.eps
            gamma = self.batch_norm.weight
            var = self.batch_norm.running_var
            scale = gamma / torch.sqrt(var + eps)
            
            # The fused kernel handles: scale * (x - mean(x))
            x = self.fused_op.fused_bn_mean_cuda(x, scale)
            
        return x