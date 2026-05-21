import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for HardSigmoid
# HardSigmoid(x) = max(0, min(1, (x + 3) / 6))
# This can be computed efficiently using fused operations to minimize memory traffic.
hardsigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hardsigmoid_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        // Compute (val + 3.0f) / 6.0f
        val = (val + 3.0f) * 0.16666667f; // 1/6 approx
        
        // Clamp between 0 and 1
        if (val < 0.0f) {
            val = 0.0f;
        } else if (val > 1.0f) {
            val = 1.0f;
        }
        
        output[idx] = val;
    }
}

torch::Tensor hardsigmoid_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    hardsigmoid_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

hardsigmoid_cpp_source = (
    "torch::Tensor hardsigmoid_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for HardSigmoid
hardsigmoid_op = load_inline(
    name="hardsigmoid_op",
    cpp_sources=hardsigmoid_cpp_source,
    cuda_sources=hardsigmoid_source,
    functions=["hardsigmoid_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a HardSigmoid activation using custom CUDA.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.hardsigmoid_op = hardsigmoid_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies HardSigmoid activation to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with HardSigmoid applied, same shape as input.
        """
        return self.hardsigmoid_op.hardsigmoid_cuda(x)