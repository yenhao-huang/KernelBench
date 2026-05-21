import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Sigmoid activation
# Sigmoid(x) = 1 / (1 + exp(-x))
sigmoid_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void sigmoid_kernel(const float* __restrict__ input, float* __restrict__ output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

torch::Tensor sigmoid_cuda_forward(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    sigmoid_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        size
    );

    return output;
}
"""

sigmoid_cpp_source = """
torch::Tensor sigmoid_cuda_forward(torch::Tensor input);
"""

# Compile the inline CUDA code
sigmoid_op = load_inline(
    name="sigmoid_op",
    cpp_sources=sigmoid_cpp_source,
    cuda_sources=sigmoid_cuda_source,
    functions=["sigmoid_cuda_forward"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a Sigmoid activation using a custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.sigmoid_cuda = sigmoid_op.sigmoid_cuda_forward
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Sigmoid activation to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape (must be on CUDA and FP32).

        Returns:
            torch.Tensor: Output tensor with Sigmoid applied, same shape as input.
        """
        if not x.is_cuda:
            return torch.sigmoid(x)
        return self.sigmoid_cuda(x)