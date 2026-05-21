import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Tanh activation
tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void tanh_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        // Compute tanh(val) using exp(2*val) for numerical stability and speed
        // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        // For large positive x, tanh(x) -> 1. For large negative x, tanh(x) -> -1.
        float exp_2x = expf(2.0f * val);
        output[idx] = (exp_2x - 1.0f) / (exp_2x + 1.0f);
    }
}

torch::Tensor tanh_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    tanh_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

tanh_cpp_source = (
    "torch::Tensor tanh_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for Tanh activation
tanh_op = load_inline(
    name="tanh_custom",
    cpp_sources=tanh_cpp_source,
    cuda_sources=tanh_source,
    functions=["tanh_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a Tanh activation using custom CUDA operator.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tanh_op = tanh_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Tanh activation to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Tanh applied, same shape as input.
        """
        return self.tanh_op.tanh_cuda(x)