import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# The optimization strategy is to fuse the element-wise operations:
# (x - subtract_value) -> HardSwish -> Mish
# Note: MaxPool2d is a reduction operation and is harder to fuse with element-wise 
# without complex tiling, so we fuse the element-wise sequence following the Conv2d.
# HardSwish(x) = x * ReLU6(x + 3) / 6
# Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))

fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ __forceinline__ float hardswish(float x) {
    float relu6 = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
    return x * relu6 / 6.0f;
}

__device__ __forceinline__ float mish(float x) {
    // mish(x) = x * tanh(softplus(x))
    // softplus(x) = log(1 + exp(x))
    // For stability: if x > 20, softplus(x) approx x, tanh(x) approx 1
    float softplus;
    if (x > 20.0f) {
        softplus = x;
    } else {
        softplus = logf(1.0f + expf(x));
    }
    return x * tanhf(softplus);
}

__global__ void fused_elementwise_kernel(const float* __restrict__ input, 
                                         float* __restrict__ output, 
                                         float subtract_val, 
                                         int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx] - subtract_val;
        val = hardswish(val);
        val = mish(val);
        output[idx] = val;
    }
}

torch::Tensor fused_ops_cuda(torch::Tensor input, float subtract_val) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_elementwise_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        subtract_val, 
        size
    );

    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor input, float subtract_val);"
)

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized Model:
    1. Conv2d remains standard (highly optimized in cuDNN).
    2. Fuses (x - subtract_value), HardSwish, and Mish into a single CUDA kernel.
    3. MaxPool2d remains standard.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = subtract_value
        self.pool = nn.MaxPool2d(pool_kernel_size)
        self.fused_ops = fused_ops

    def forward(self, x):
        # Step 1: Convolution
        x = self.conv(x)
        
        # Step 2: Fused (Subtract -> HardSwish -> Mish)
        # We pass the subtract_value as a float to the kernel
        x = self.fused_ops.fused_ops_cuda(x, float(self.subtract_value))
        
        # Step 3: MaxPool
        x = self.pool(x)
        
        return x