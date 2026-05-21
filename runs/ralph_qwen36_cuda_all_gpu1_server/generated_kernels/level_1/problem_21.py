import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Sigmoid activation
sigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void sigmoid_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        // Numerically stable sigmoid: 1 / (1 + exp(-x))
        // For large positive x, exp(-x) is small.
        // For large negative x, we can use exp(x) / (1 + exp(x)) to avoid overflow in denominator if needed, 
        // but standard formula usually handles it well enough or we clamp.
        // A common stable implementation:
        float exp_val;
        if (val >= 0) {
            exp_val = exp(-val);
            output[idx] = 1.0f / (1.0f + exp_val);
        } else {
            exp_val = exp(val);
            output[idx] = exp_val / (1.0f + exp_val);
        }
    }
}

torch::Tensor sigmoid_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    sigmoid_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

sigmoid_cpp_source = (
    "torch::Tensor sigmoid_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for Sigmoid
sigmoid_op = load_inline(
    name="sigmoid_custom",
    cpp_sources=sigmoid_cpp_source,
    cuda_sources=sigmoid_source,
    functions=["sigmoid_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a Sigmoid activation using custom CUDA operator.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.sigmoid_op = sigmoid_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Sigmoid activation to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Sigmoid applied, same shape as input.
        """
        return self.sigmoid_op.sigmoid_cuda(x)