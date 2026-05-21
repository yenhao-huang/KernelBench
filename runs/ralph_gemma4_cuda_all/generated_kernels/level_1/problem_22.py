import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Tanh activation
# Tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
# Using __expf for fast hardware-accelerated exponential in FP32
tanh_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void tanh_kernel(const float* __restrict__ x, float* __restrict__ out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = tanhf(x[idx]);
    }
}

torch::Tensor tanh_cuda_forward(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    tanh_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), 
        out.data_ptr<float>(), 
        size
    );

    return out;
}
"""

tanh_cpp_source = """
torch::Tensor tanh_cuda_forward(torch::Tensor x);
"""

# Compile the inline CUDA code
tanh_op = load_inline(
    name="tanh_op",
    cpp_sources=tanh_cpp_source,
    cuda_sources=tanh_cuda_source,
    functions=["tanh_cuda_forward"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a Tanh activation using a custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tanh_cuda = tanh_op.tanh_cuda_forward
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Tanh activation to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape (must be on CUDA and FP32).

        Returns:
            torch.Tensor: Output tensor with Tanh applied, same shape as input.
        """
        if not x.is_cuda:
            return torch.tanh(x)
        return self.tanh_cuda(x)