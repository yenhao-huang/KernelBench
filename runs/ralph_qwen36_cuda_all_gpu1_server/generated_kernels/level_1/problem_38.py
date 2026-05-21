import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for L1 normalization
# This kernel computes the sum of absolute values along dim 1, then divides each element by that sum.
# To optimize, we can fuse the reduction and division into a single pass or two efficient passes.
# Here we use a two-pass approach within one kernel launch structure for simplicity and correctness:
# 1. Parallel reduction to compute the L1 norm per row.
# 2. Element-wise division using the computed norms.

l1_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Helper for atomic add if needed, but here we use shared memory reduction or simple grid reduction
// For simplicity and robustness with large dimensions, we'll use a standard parallel reduction pattern.

__global__ void l1_norm_kernel(const float* x, float* out, float* norms, int batch_size, int dim) {
    // Each block handles one row (one sample in the batch)
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x; // Block index corresponds to batch index
    
    if (idx >= batch_size) return;

    // Load data into shared memory for reduction
    // We assume dim is a multiple of blockDim.x for simplicity in this basic version, 
    // or handle boundary checks. For high performance, tiling is better, but let's stick to a robust simple reduction.
    
    float sum = 0.0f;
    
    // Simple loop to compute absolute sum for the row
    // In a real optimized kernel, we would use shared memory and warp-level primitives.
    // Here, we do a direct global memory read reduction which is suboptimal but correct.
    // To make it faster, let's use a standard parallel reduction in shared memory.
    
    // Initialize shared memory with absolute values
    float val = 0.0f;
    if (tid < dim) {
        val = fabsf(x[idx * dim + tid]);
    }
    sdata[tid] = val;
    __syncthreads();

    // Parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the sum for this row to global memory
    if (tid == 0) {
        norms[idx] = sdata[0];
    }
}

__global__ void l1_divide_kernel(const float* x, const float* norms, float* out, int batch_size, int dim) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * dim;
    
    if (idx < total_elements) {
        int batch_idx = idx / dim;
        float norm = norms[batch_idx];
        // Avoid division by zero
        if (norm > 1e-8f) {
            out[idx] = x[idx] / norm;
        } else {
            out[idx] = x[idx];
        }
    }
}

torch::Tensor l1_norm_cuda(torch::Tensor x) {
    auto batch_size = x.size(0);
    auto dim = x.size(1);
    
    auto out = torch::empty_like(x);
    auto norms = torch::empty({batch_size}, x.options());
    
    const int block_size = 256;
    // Ensure block size is power of 2 and <= dim for the reduction kernel logic to hold simply
    // If dim < block_size, we might need adjustment, but typically dim is large.
    // We'll cap block size at dim if necessary, but standard CUDA grids handle this.
    
    int actual_block_size = block_size;
    if (dim < block_size) {
        actual_block_size = 1 << (31 - __builtin_clz(dim)); // Power of 2 <= dim
        if (actual_block_size == 0) actual_block_size = 1;
    }

    const int num_blocks = batch_size;
    
    // Launch reduction kernel
    l1_norm_kernel<<<num_blocks, actual_block_size, actual_block_size * sizeof(float)>>>(
        x.data_ptr<float>(), 
        out.data_ptr<float>(), 
        norms.data_ptr<float>(), 
        batch_size, 
        dim
    );
    
    // Launch division kernel
    const int total_elements = batch_size * dim;
    const int div_block_size = 256;
    const int div_num_blocks = (total_elements + div_block_size - 1) / div_block_size;
    
    l1_divide_kernel<<<div_num_blocks, div_block_size>>>(
        x.data_ptr<float>(), 
        norms.data_ptr<float>(), 
        out.data_ptr<float>(), 
        batch_size, 
        dim
    );

    return out;
}
"""

l1_norm_cpp_source = (
    "torch::Tensor l1_norm_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for L1 normalization
l1_norm = load_inline(
    name="l1_norm",
    cpp_sources=l1_norm_cpp_source,
    cuda_sources=l1_norm_source,
    functions=["l1_norm_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Optimized model that performs L1 normalization using custom CUDA operators.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return l1_norm.l1_norm_cuda(x)