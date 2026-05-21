import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for L2 normalization
# This kernel computes the L2 norm along dimension 1 and normalizes the input in a fused manner.
# It uses shared memory for efficient reduction if possible, but for simplicity and robustness with large dims,
# we use a two-pass approach or a single pass with atomic adds if needed. 
# However, for a single kernel optimization, a common pattern is:
# 1. Compute sum of squares per row (reduction).
# 2. Normalize each element by the computed norm.
# To do this in one kernel launch efficiently without shared memory limitations for large dims, 
# we can use a grid-stride loop for the reduction and then another pass, or fuse them.
# Given the constraint of "inline" and simplicity, a two-kernel approach (norm + div) is often cleaner,
# but let's try to do it in one kernel if possible or just optimize the norm calculation.
# Actually, PyTorch's torch.norm is already quite optimized. The bottleneck might be the division.
# Let's create a custom kernel that does: out = x / sqrt(sum(x^2))
# We can compute the sum of squares in a reduction kernel, then use it for normalization.
# Or, we can do it in one kernel using shared memory for small blocks, but dim=65535 is large.
# A better approach for large dims: 
# 1. Kernel 1: Compute sum of squares per row (output shape [batch_size]).
# 2. Kernel 2: Normalize x by the precomputed norms.

l2_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel to compute sum of squares along dimension 1 for each row
__global__ void sum_squares_kernel(const float* x, float* sum_sq, int batch_size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        float sum = 0.0f;
        const float* row_ptr = x + idx * dim;
        for (int i = 0; i < dim; ++i) {
            float val = row_ptr[i];
            sum += val * val;
        }
        sum_sq[idx] = sum;
    }
}

// Kernel to normalize the tensor using precomputed norms
__global__ void normalize_kernel(const float* x, const float* sum_sq, float* out, int batch_size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * dim) {
        int row = idx / dim;
        int col = idx % dim;
        float norm = sqrtf(sum_sq[row]);
        // Avoid division by zero
        if (norm > 1e-8f) {
            out[idx] = x[idx] / norm;
        } else {
            out[idx] = 0.0f;
        }
    }
}

torch::Tensor l2_norm_cuda(torch::Tensor x) {
    auto batch_size = x.size(0);
    auto dim = x.size(1);
    
    // Allocate output for sum of squares
    torch::Tensor sum_sq = torch::empty({batch_size}, x.options());
    
    const int block_size = 256;
    const int num_blocks_sum = (batch_size + block_size - 1) / block_size;
    
    sum_squares_kernel<<<num_blocks_sum, block_size>>>(x.data_ptr<float>(), sum_sq.data_ptr<float>(), batch_size, dim);
    
    // Allocate output tensor
    torch::Tensor out = torch::empty_like(x);
    
    const int total_elements = batch_size * dim;
    const int num_blocks_norm = (total_elements + block_size - 1) / block_size;
    
    normalize_kernel<<<num_blocks_norm, block_size>>>(x.data_ptr<float>(), sum_sq.data_ptr<float>(), out.data_ptr<float>(), batch_size, dim);
    
    return out;
}
"""

l2_norm_cpp_source = (
    "torch::Tensor l2_norm_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for L2 normalization
l2_norm = load_inline(
    name="l2_norm",
    cpp_sources=l2_norm_cpp_source,
    cuda_sources=l2_norm_source,
    functions=["l2_norm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs L2 normalization using custom CUDA operators.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return l2_norm.l2_norm_cuda(x)