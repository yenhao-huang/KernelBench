import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Instance Normalization
# This implementation fuses mean/variance calculation, normalization, and scaling/shift (if needed)
# into a single pass or minimal passes to maximize memory bandwidth efficiency.
instance_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Helper to get the number of elements per instance
// For InstanceNorm2d, normalization is over H and W for each channel in each batch item.
// Shape: (N, C, H, W)
// Elements per instance = H * W

__global__ void instance_norm_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float* __restrict__ mean,      // Output mean for each (N, C) pair
    float* __restrict__ var,        // Output variance for each (N, C) pair
    const float* __restrict__ weight,   // Gamma (optional, can be null if not used, but IN2d usually has it)
    const float* __restrict__ bias,     // Beta (optional)
    int N, int C, int H, int W,
    float eps
) {
    // Each thread block handles one instance (one N, one C)
    // We launch N*C blocks.
    
    int idx = blockIdx.x;
    if (idx >= N * C) return;

    int n = idx / C;
    int c = idx % C;

    const float* input_ptr = input + idx * H * W;
    float* output_ptr = output + idx * H * W;
    
    // Calculate mean and variance for this instance
    float sum = 0.0f;
    float sum_sq = 0.0f;
    int count = H * W;

    // First pass: compute mean and variance
    // Using shared memory could optimize, but for simplicity and correctness with large H*W, 
    // direct global access is often sufficient if we ensure coalescing. 
    // However, to be safe and fast, let's do a simple loop.
    
    for (int i = 0; i < count; ++i) {
        float val = input_ptr[i];
        sum += val;
        sum_sq += val * val;
    }

    float m = sum / count;
    float v = sum_sq / count - m * m;
    
    // Add epsilon for numerical stability
    v += eps;
    
    // Store mean and variance
    mean[idx] = m;
    var[idx] = v;

    // Second pass: normalize and apply affine transform if present
    float inv_std = rsqrtf(v);
    
    // If weight/bias are provided, they are 1D tensors of size C.
    // We need to access them correctly.
    const float* w_ptr = weight ? (weight + c) : nullptr;
    const float* b_ptr = bias ? (bias + c) : nullptr;

    for (int i = 0; i < count; ++i) {
        float val = input_ptr[i];
        float normalized = (val - m) * inv_std;
        
        if (w_ptr && b_ptr) {
            output_ptr[i] = normalized * w_ptr[0] + b_ptr[0];
        } else {
            output_ptr[i] = normalized;
        }
    }
}

torch::Tensor instance_norm_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, float eps) {
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(x.dim() == 4, "Input must be a 4D tensor (N, C, H, W)");
    
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    auto output = torch::empty_like(x);
    // We need to store mean and variance for backward pass, but here we just return the normalized output.
    // The problem asks for forward optimization. 
    // Note: Standard nn.InstanceNorm2d returns (output, running_mean, running_var) if track_running_stats is true,
    // or just output if not. For inference/forward only, we just need the normalized tensor.
    
    const int total_instances = N * C;
    const int block_size = 1; // One thread per instance for simplicity in this fused kernel
    
    // Launch kernel
    instance_norm_kernel<<<total_instances, block_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        nullptr, // mean output not strictly needed for forward-only inference unless requested
        nullptr, // var output not strictly needed
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        N, C, H, W, eps
    );

    cudaDeviceSynchronize();
    
    return output;
}
"""

instance_norm_cpp_source = (
    "torch::Tensor instance_norm_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, float eps);"
)

# Compile the inline CUDA code for Instance Normalization
instance_norm_module = load_inline(
    name="instance_norm_cuda",
    cpp_sources=instance_norm_cpp_source,
    cuda_sources=instance_norm_source,
    functions=["instance_norm_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Optimized model that performs Instance Normalization using a custom CUDA operator.
    """
    def __init__(self, num_features: int):
        """
        Initializes the InstanceNorm layer with custom CUDA implementation.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(ModelNew, self).__init__()
        # We keep the parameters for weight and bias as they are part of the standard InstanceNorm2d interface
        # However, since we are replacing the operator, we can handle them directly in the CUDA kernel.
        # To maintain compatibility with the original model's parameter structure if needed, 
        # we could store them, but for pure forward speedup, we just pass them to the custom op.
        # The original nn.InstanceNorm2d has learnable weight and bias by default.
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = 1e-5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Instance Normalization to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output tensor with Instance Normalization applied, same shape as input.
        """
        return instance_norm_module.instance_norm_cuda(x, self.weight, self.bias, self.eps)