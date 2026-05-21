import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# The optimization strategy:
# The GEMM (Linear layer) is best left to cuBLAS for maximum performance.
# However, the subsequent operations: GroupNorm -> Swish -> Multiply -> Swish
# are all element-wise or reduction-based operations that can be fused.
# We will fuse GroupNorm, the first Swish, the Multiply, and the second Swish 
# into a single CUDA kernel to minimize memory bandwidth bottlenecks.

fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Helper for sigmoid
__device__ __forceinline__ float sigmoid_f(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Helper for swish
__device__ __forceinline__ float swish_f(float x) {
    return x * sigmoid_f(x);
}

__global__ void fused_norm_swish_multiply_swish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ mul_weight,
    float* __restrict__ output,
    int batch_size,
    int out_features,
    int num_groups,
    float eps) 
{
    // Each thread handles one element (batch_idx, feature_idx)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_features;
    
    if (idx >= total_elements) return;

    int b = idx / out_features;
    int f = idx % out_features;

    // GroupNorm logic:
    // Group index for this feature
    int g = f / (out_features / num_groups);
    int group_size = out_features / num_groups;
    int group_start_f = g * group_size;

    // Note: In a real high-performance kernel, we would use shared memory 
    // to compute mean and variance for the group. 
    // For simplicity and correctness in this inline example, we assume 
    // the mean and variance are pre-calculated or we use a two-pass approach.
    // However, to keep this kernel single-pass and efficient, we'll assume 
    // the user provides the mean and variance or we compute them.
    // To make this a truly "fused" kernel that replaces the sequence, 
    // we'll implement the element-wise part assuming mean/var are passed.
}

// Since computing GroupNorm mean/var requires a reduction, 
// a single-pass element-wise kernel is most efficient if we pass mean/var.
// But to replace the whole block, we'll implement a kernel that 
// performs the element-wise transformations after the GroupNorm 
// has been applied, or we can fuse the element-wise parts.

// Let's optimize the sequence: x = GN(x); x = swish(x); x = x * mul; x = swish(x)
// This is a massive memory bandwidth saver.

__global__ void fused_elementwise_kernel(
    float* x,
    const float* mul_weight,
    int size) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        // First Swish: x * sigmoid(x)
        val = val * (1.0f / (1.0f + expf(-val)));
        // Multiply
        val = val * mul_weight[idx % (size / (x / mul_weight).numel())]; // This is wrong logic for indexing
        // Let's fix the indexing in the actual implementation below.
    }
}
"""

# Re-writing a cleaner, more robust fused kernel
# We will fuse: x = x * sigmoid(x) -> x = x * mul_weight -> x = x * sigmoid(x)
# This avoids 3 separate read/write passes.

fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_swish_mul_swish_kernel(
    float* x,
    const float* mul_weight,
    int batch_size,
    int out_features,
    int total_elements) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        int f = idx % out_features;
        float val = x[idx];
        
        // 1. First Swish: x * sigmoid(x)
        val = val * (1.0f / (1.0f + expf(-val)));
        
        // 2. Multiply by weight
        val = val * mul_weight[f];
        
        // 3. Second Swish: x * sigmoid(x)
        val = val * (1.0f / (1.0f + expf(-val)));
        
        x[idx] = val;
    }
}

void launch_fused_kernel(torch::Tensor x, torch::Tensor mul_weight, int batch_size, int out_features) {
    int total_elements = x.numel();
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_swish_mul_swish_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        mul_weight.data_ptr<float>(),
        batch_size,
        out_features,
        total_elements
    );
}
"""

fused_kernel_cpp_source = "void launch_fused_kernel(torch::Tensor x, torch::Tensor mul_weight, int batch_size, int out_features);"

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["launch_fused_kernel"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))
        self.fused_ops = fused_ops

    def forward(self, x):
        # 1. GEMM
        x = self.gemm(x)
        
        # 2. GroupNorm (Standard PyTorch is highly optimized)
        x = self.group_norm(x)
        
        # 3. Fused Swish -> Multiply -> Swish
        # We perform this in-place to save memory
        batch_size = x.size(0)
        self.fused_ops.launch_fused_kernel(x, self.multiply_weight, batch_size, self.out_features)
        
        return x