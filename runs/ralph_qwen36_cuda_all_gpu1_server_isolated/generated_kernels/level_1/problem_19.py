import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ReLU activation
relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        output[idx] = val > 0.0f ? val : 0.0f;
    }
}

torch::Tensor relu_cuda(torch::Tensor x) {
    auto out = torch::empty_like(x);
    auto size = x.numel();

    if (size == 0) {
        return out;
    }

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    relu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    // Synchronize to ensure kernel completion before returning
    cudaDeviceSynchronize();

    return out;
}
"""

relu_cpp_source = (
    "torch::Tensor relu_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for ReLU
relu_module = load_inline(
    name="relu_custom",
    cpp_sources=relu_cpp_source,
    cuda_sources=relu_source,
    functions=["relu_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a ReLU activation using a custom CUDA operator.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.relu_op = relu_module.relu_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies ReLU activation to the input tensor using a custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with ReLU applied, same shape as input.
        """
        return self.relu_op(x)