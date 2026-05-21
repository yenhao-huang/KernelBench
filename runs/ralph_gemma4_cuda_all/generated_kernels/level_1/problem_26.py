import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for GELU activation
# GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
# Using the fast approximation or the exact erf implementation.
# Here we implement the exact version using the standard math library.
gelu_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel(const float* __restrict__ input, float* __restrict__ output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        // Exact GELU: 0.5 * x * (1.0 + erf(x / sqrt(2.0)))
        output[idx] = 0.5f * x * (1.0f + erff(x * 0.70710678118f));
    }
}

torch::Tensor gelu_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    gelu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        size
    );

    return output;
}
"""

gelu_cpp_source = """
torch::Tensor gelu_cuda(torch::Tensor input);
"""

# Compile the inline CUDA code
gelu_extension = load_inline(
    name="gelu_extension",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_cuda_source,
    functions=["gelu_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a GELU activation using a custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gelu_cuda = gelu_extension.gelu_cuda
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies GELU activation to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape, must be on CUDA and FP32.

        Returns:
            torch.Tensor: Output tensor with GELU applied, same shape as input.
        """
        if not x.is_cuda:
            return torch.nn.functional.gelu(x)
        
        # Ensure input is contiguous for the kernel
        if not x.is_contiguous():
            x = x.contiguous()
            
        return self.gelu_cuda(x)