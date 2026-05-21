import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for fused subtract + Mish activation
fused_subtract_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_subtract_mish_kernel(float* data, int size, float sub1, float sub2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = data[idx] - sub1 - sub2;
        // Mish activation: x * tanh(softplus(x))
        float sp;
        if (x > 20.0f) {
            sp = x;
        } else {
            sp = log1pf(expf(x));
        }
        float tanh_sp = tanhf(sp);
        data[idx] = x * tanh_sp;
    }
}

torch::Tensor fused_subtract_mish_cuda(torch::Tensor input, float sub1, float sub2) {
    // Ensure input is contiguous and on CUDA
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    
    auto size = input.numel();
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    // In-place operation: modify input directly
    fused_subtract_mish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), size, sub1, sub2);
    
    return input;
}
"""

fused_subtract_mish_cpp_source = """
torch::Tensor fused_subtract_mish_cuda(torch::Tensor input, float sub1, float sub2);
"""

# Compile the inline CUDA code
fused_subtract_mish = load_inline(
    name="fused_subtract_mish",
    cpp_sources=fused_subtract_mish_cpp_source,
    cuda_sources=fused_subtract_mish_source,
    functions=["fused_subtract_mish_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Model that performs a convolution, subtracts two values, applies Mish activation.
    Uses a fused CUDA kernel for subtract + Mish.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
        self.fused_subtract_mish = fused_subtract_mish

    def forward(self, x):
        x = self.conv(x)
        # Fused subtract and Mish in-place
        x = self.fused_subtract_mish.fused_subtract_mish_cuda(
            x.contiguous(), self.subtract_value_1, self.subtract_value_2
        )
        return x