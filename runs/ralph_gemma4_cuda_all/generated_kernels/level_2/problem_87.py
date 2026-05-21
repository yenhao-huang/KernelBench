import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused-elementwise-operations (subtraction and Mish)
fused_elementwise_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ inline float mish_device(float x) {
    // Mish(x) = x * tanh(softplus(x))
    // softplus(x) = log(1 + exp(x))
    return x * std::tanh(std::log1p(std::exp(x)));
}

__global__ void fused_sub_mish_kernel(const float* __restrict__ input, 
                                      float* __restrict__ output, 
                                      float val1, 
                                      float val2, 
                                      int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx] - val1 - val2;
        output[idx] = mish_device(x);
    }
}

torch::Tensor fused_sub_mish_cuda(torch::Tensor input, float val1, float val2) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_sub_mish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        val1, 
        val2, 
        size
    );

    return output;
}
"""

fused_elementwise_cpp_source = (
    "torch::Tensor fused_sub_mish_cuda(torch::Tensor input, float val1, float val2);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_elementwise_cpp_source,
    cuda_sources=fused_elementwise_source,
    functions=["fused_sub_mish_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        # Fuse: x - val1 - val2 and Mish(x)
        x = self.fused_ops.fused_sub_mish_cuda(x, self.subtract_value_1, self.subtract_value_2)
        return x