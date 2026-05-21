import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Swish activation (x * sigmoid(x))
# We use a single kernel to perform the multiplication and sigmoid in one pass,
# reducing memory bandwidth requirements (fusing the operations).
swish_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void swish_kernel(const float* __restrict__ x, float* __restrict__ out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        // sigmoid(x) = 1 / (1 + exp(-x))
        float sigmoid_val = 1.0f / (1.0f + expf(-val));
        out[idx] = val * sigmoid_val;
    }
}

torch::Tensor swish_cuda_forward(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    swish_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), 
        out.data_ptr<float>(), 
        size
    );

    return out;
}
"""

swish_cpp_source = """
torch::Tensor swish_cuda_forward(torch::Tensor x);
"""

# Compile the inline CUDA code
swish_extension = load_inline(
    name="swish_extension",
    cpp_sources=swish_cpp_source,
    cuda_sources=swish_cuda_source,
    functions=["swish_cuda_forward"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a Swish activation using a custom fused CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.swish_cuda = swish_extension.swish_cuda_forward
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Swish activation to the input tensor using the custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape, must be on CUDA and FP32.

        Returns:
            torch.Tensor: Output tensor with Swish applied, same shape as input.
        """
        if not x.is_cuda:
            return x * torch.sigmoid(x)
        
        # Ensure the tensor is contiguous for the CUDA kernel
        if not x.is_contiguous():
            x = x.contiguous()
            
        return self.swish_cuda(x)