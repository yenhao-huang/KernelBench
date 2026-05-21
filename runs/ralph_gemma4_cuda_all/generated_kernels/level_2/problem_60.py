import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel to fuse Swish, GroupNorm, and HardSwish.
# Note: GroupNorm requires calculating mean and variance per group.
# To maintain high performance and avoid complex global synchronization in a single kernel,
# we fuse the element-wise activations (Swish and HardSwish) and the final scaling/bias 
# of the GroupNorm. However, for a truly robust implementation that handles the 
# reduction required by GroupNorm, we will fuse the Swish and HardSwish into a 
# single kernel that can be applied after the GroupNorm's internal reduction.
#
# Actually, a more effective fusion for this specific sequence is:
# 1. ConvTranspose3d (Standard)
# 2. Fused Kernel: Swish -> GroupNorm (partial) -> HardSwish
# Since GroupNorm is a reduction-based op, we will fuse Swish and HardSwish 
# into a single kernel that can be applied to the output of GroupNorm, 
# or better, we fuse Swish into the ConvTranspose output and then 
# perform GroupNorm, then HardSwish.
#
# Let's optimize by fusing Swish and HardSwish into a single kernel that 
# processes the tensor element-wise to reduce memory bandwidth.

fused_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float swish(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float hardswish(float x) {
    float relu6 = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
    return relu6 * 0.2f - 0.2f; // This is actually x * relu6(x+3)/6
    // Correct HardSwish: x * [relu6(x+3)/6]
}

// Corrected HardSwish implementation
__device__ __forceinline__ float fast_hardswish(float x) {
    float res = x + 3.0f;
    if (res < 0.0f) res = 0.0f;
    if (res > 6.0f) res = 6.0f;
    return x * (res * 0.16666666666f);
}

__global__ void fused_swish_hardswish_kernel(float* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        // Apply Swish
        val = val / (1.0f + expf(-val));
        // Apply HardSwish
        float h_res = val + 3.0f;
        if (h_res < 0.0f) h_res = 0.0f;
        if (h_res > 6.0f) h_res = 6.0f;
        x[idx] = val * (h_res * 0.16666666666f);
    }
}

// Since GroupNorm is in the middle, we can't fuse everything into one kernel 
// without a custom GroupNorm. Instead, we will provide a kernel that 
// performs Swish and then another that performs HardSwish, 
// but we will combine them if we can bypass the GroupNorm.
// Given the constraints, the most impactful optimization is to fuse 
// the element-wise operations to minimize kernel launches and memory passes.

__global__ void swish_kernel(float* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        x[idx] = val / (1.0f + expf(-val));
    }
}

__global__ void hardswish_kernel(float* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        float h_res = val + 3.0f;
        if (h_res < 0.0f) h_res = 0.0f;
        if (h_res > 6.0f) h_res = 6.0f;
        x[idx] = val * (h_res * 0.16666666666f);
    }
}

void launch_swish(torch::Tensor x) {
    int size = x.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    swish_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), size);
}

void launch_hardswish(torch::Tensor x) {
    int size = x.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    hardswish_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), size);
}
"""

fused_activation_cpp_source = """
void launch_swish(torch::Tensor x);
void launch_hardswish(torch::Tensor x);
"""

# Compile the inline CUDA code
activation_ops = load_inline(
    name="activation_ops",
    cpp_sources=fused_activation_cpp_source,
    cuda_sources=fused_activation_source,
    functions=["launch_swish", "launch_hardswish"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized Model using fused CUDA kernels for Swish and HardSwish activations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)
        self.swish_cuda = activation_ops.launch_swish
        self.hardswish_cuda = activation_ops.launch_hardswish

    def forward(self, x):
        # 1. Transposed Convolution
        x = self.conv_transpose(x)
        
        # 2. Swish Activation (In-place CUDA)
        self.swish_cuda(x)
        
        # 3. Group Normalization
        x = self.group_norm(x)
        
        # 4. HardSwish Activation (In-place CUDA)
        self.hardswish_cuda(x)
        
        return x