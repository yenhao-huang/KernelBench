import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ELU activation
# ELU(x) = x if x > 0 else alpha * (exp(x) - 1)
elu_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void elu_kernel(const float* __restrict__ input, float* __restrict__ output, float alpha, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        if (val > 0.0f) {
            output[idx] = val;
        } else {
            output[idx] = alpha * (expf(val) - 1.0f);
        }
    }
}

torch::Tensor elu_cuda_forward(torch::Tensor input, float alpha) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        alpha, 
        size
    );

    return output;
}
"""

elu_cpp_source = """
torch::Tensor elu_cuda_forward(torch::Tensor input, float alpha);
"""

# Compile the inline CUDA code
elu_extension = load_inline(
    name="elu_extension",
    cpp_sources=elu_cpp_source,
    cuda_sources=elu_cuda_source,
    functions=["elu_cuda_forward"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs an ELU activation using a custom CUDA kernel.
    """
    def __init__(self, alpha: float = 1.0):
        """
        Initializes the ELU model.

        Args:
            alpha (float, optional): The alpha parameter for the ELU function. Defaults to 1.0.
        """
        super(ModelNew, self).__init__()
        self.alpha = alpha
        self.elu_cuda = elu_extension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies ELU activation to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape. Must be on CUDA and FP32.

        Returns:
            torch.Tensor: Output tensor with ELU applied, same shape as input.
        """
        if not x.is_cuda:
            return F.elu(x, alpha=self.alpha)
        
        # Ensure input is float32 for the kernel
        if x.dtype != torch.float32:
            return F.elu(x, alpha=self.alpha)

        return self.elu_cuda.elu_cuda_forward(x, self.alpha)