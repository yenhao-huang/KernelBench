import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Batch Normalization (Inference Mode)
# This implementation fuses the normalization steps:
# 1. Subtract mean
# 2. Divide by sqrt(variance + eps)
# 3. Multiply by gamma
# 4. Add beta
# It assumes running_mean and running_var are already computed and stored in the model state.

bn_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void batch_norm_kernel(
    const float* input, 
    const float* mean, 
    const float* var, 
    const float* gamma, 
    const float* beta, 
    float* output, 
    int num_features, 
    int spatial_size, 
    float eps) 
{
    // Each thread handles one element in the batch for a specific feature channel
    // Total elements = batch_size * num_features * spatial_size
    // We iterate over all elements.
    
    int total_elements = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (idx < total_elements) {
        // Calculate indices: [n, c, h, w] flattened
        // spatial_size = H * W
        // input shape: [N, C, H, W] -> [N, C, spatial_size]
        
        int n = idx / (num_features * spatial_size);
        int remainder = idx % (num_features * spatial_size);
        int c = remainder / spatial_size;
        int s = remainder % spatial_size; // spatial index
        
        float x = input[idx];
        
        // Get mean and var for this channel. 
        // Mean and Var are 1D tensors of size [num_features]
        float mu = mean[c];
        float sigma = var[c];
        
        // Normalize: (x - mu) / sqrt(sigma + eps)
        float inv_std = rsqrtf(sigma + eps);
        float normalized = (x - mu) * inv_std;
        
        // Scale and shift: gamma * normalized + beta
        float g = gamma[c];
        float b = beta[c];
        
        output[idx] = g * normalized + b;
        
        idx += total_elements;
    }
}

torch::Tensor batch_norm_cuda(
    torch::Tensor input, 
    torch::Tensor running_mean, 
    torch::Tensor running_var, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    float eps) 
{
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(running_mean.is_cuda(), "Running mean must be on CUDA");
    TORCH_CHECK(running_var.is_cuda(), "Running var must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight must be on CUDA");
    TORCH_CHECK(bias.is_cuda(), "Bias must be on CUDA");

    auto num_features = input.size(1);
    int spatial_size = 1;
    for (int i = 2; i < input.dim(); ++i) {
        spatial_size *= input.size(i);
    }
    
    // Total number of elements to process
    int total_elements = input.numel();
    
    auto output = torch::empty_like(input);

    if (total_elements == 0) {
        return output;
    }

    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    batch_norm_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        num_features,
        spatial_size,
        eps
    );

    return output;
}
"""

bn_cpp_source = (
    "torch::Tensor batch_norm_cuda("
    "torch::Tensor input, "
    "torch::Tensor running_mean, "
    "torch::Tensor running_var, "
    "torch::Tensor weight, "
    "torch::Tensor bias, "
    "float eps);"
);

# Compile the inline CUDA code
batch_norm_lib = load_inline(
    name="batch_norm_cuda",
    cpp_sources=bn_cpp_source,
    cuda_sources=bn_source,
    functions=["batch_norm_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Optimized model that performs Batch Normalization using a custom CUDA kernel.
    """
    def __init__(self, num_features: int):
        """
        Initializes the BatchNorm layer with custom CUDA implementation.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(ModelNew, self).__init__()
        
        # We need to replicate the parameters of nn.BatchNorm2d manually 
        # because we are bypassing the standard forward pass.
        # BatchNorm2d has: weight (gamma), bias (beta), running_mean, running_var
        
        self.register_parameter('weight', nn.Parameter(torch.ones(num_features)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(num_features)))
        
        # Register buffers for running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
        # BatchNorm2d default eps is 1e-5
        self.eps = 1e-5
        
        # Track that we are in eval mode for inference (no gradient tracking for stats)
        self.training = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Batch Normalization to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Batch Normalization applied, same shape as input.
        """
        # Ensure tensors are on the correct device and dtype
        if not x.is_cuda:
            raise RuntimeError("Input must be on CUDA for this optimized model.")
            
        # Use the custom CUDA function
        out = batch_norm_lib.batch_norm_cuda(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.eps
        )
        
        return out


def get_inputs():
    # randomly generate input tensors based on the model architecture
    x = torch.rand(batch_size, features, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [features]