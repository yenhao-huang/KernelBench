import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel that fuses Matmul, Divide, Sum, and Scale.
# This avoids multiple memory allocations and transfers between kernels.
# We assume input x is (batch_size, input_size) and weight is (hidden_size, input_size).
# The operation is: output = (sum(x @ W.T) / 2) * scaling_factor
# Which simplifies to: output = sum(x @ W.T) * (scaling_factor / 2)

custom_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void fused_gemm_sum_scale_kernel(
    const float* __restrict__ x,      // [batch_size, input_size]
    const float* __restrict__ weight, // [hidden_size, input_size]
    float* __restrict__ out,          // [batch_size, 1]
    int batch_size,
    int input_size,
    int hidden_size,
    float scale_factor
) {
    // Each block handles one row of the output (one sample in the batch)
    // However, for large matrices, we might want to parallelize within the row too.
    // Given the dimensions (1024 x 8192), a simple approach where each thread block 
    // computes the dot product for one output element is feasible if we optimize memory access.
    
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    // We will compute the sum of products for this batch item against all hidden units?
    // No, the original code does: x @ W.T -> [batch, hidden]. Then sum(dim=1) -> [batch, 1].
    // So we only need to compute ONE scalar per batch item.
    
    float sum_val = 0.0f;
    
    // Load weights and input into shared memory or just access global memory efficiently.
    // Since input_size is large (8192), we iterate over it.
    // To optimize, we can use a simple loop. For better performance, one might tile, 
    // but for a single scalar reduction per thread block, the overhead of complex tiling 
    // might not be worth it unless we parallelize across hidden units (which we don't have here).
    
    // Let's optimize by ensuring coalesced access if possible. 
    // x is [batch, input_size]. weight is [hidden, input_size].
    // We are computing dot(x[batch_idx], weight[:, input_size]).
    // Wait, the original code: torch.matmul(x, self.weight.T).
    // x: (B, I), W: (H, I). W.T: (I, H).
    // Result: (B, H).
    // Then sum(dim=1): Sum over H. Result: (B, 1).
    
    // So for a specific batch item 'b', we need to compute:
    // sum_{h=0}^{H-1} ( sum_{i=0}^{I-1} x[b, i] * weight[h, i] )
    // This can be rewritten as:
    // sum_{i=0}^{I-1} x[b, i] * ( sum_{h=0}^{H-1} weight[h, i] )
    
    // Let's precompute column sums of weight? No, we can't do that in the kernel easily without extra pass.
    // Alternatively, we can just compute it directly.
    // Direct computation:
    // For each batch item b:
    //   total_sum = 0
    //   for h in range(H):
    //     dot = 0
    //     for i in range(I):
    //       dot += x[b, i] * weight[h, i]
    //     total_sum += dot
    
    // This is O(B * H * I). With B=1024, H=8192, I=8192, this is huge. 
    // 1024 * 8192 * 8192 ~ 6.8e10 operations.
    
    // Let's stick to the standard GEMM structure but fused with reduction.
    // We assign one thread block per output element (batch, hidden).
    // But we only need the SUM over hidden.
    
    // Optimized approach:
    // 1. Compute the full GEMM result in shared memory or registers if possible? No, H is too big.
    // 2. Use a parallel reduction strategy.
    
    // Let's use one thread block per batch item.
    // Each thread in the block handles a subset of the hidden units.
    // But we need to sum over ALL hidden units for that batch item.
    
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    float local_sum = 0.0f;
    
    // We iterate over input_size (I) and hidden_size (H).
    // To maximize parallelism, let's have each thread compute a partial sum for a subset of (i, h) pairs?
    // Or simpler: Each thread computes the dot product for one specific hidden unit h.
    // Then we reduce across threads to get the sum over H.
    
    // Number of hidden units per thread? 
    // If we have 1024 threads, and H=8192, each thread handles 8 hidden units.
    
    int start_h = tid * (hidden_size / num_threads);
    int end_h = (tid + 1) * (hidden_size / num_threads);
    
    // Handle remainder if hidden_size is not divisible by num_threads
    if (tid == num_threads - 1) {
        end_h = hidden_size;
    }

    for (int h = start_h; h < end_h; ++h) {
        float dot_product = 0.0f;
        // Access x[b, i] and weight[h, i]
        // x is row-major: x[b * input_size + i]
        // weight is row-major: weight[h * input_size + i]
        
        // Loop unrolling or simple loop for I=8192
        for (int i = 0; i < input_size; ++i) {
            dot_product += x[batch_idx * input_size + i] * weight[h * input_size + i];
        }
        local_sum += dot_product;
    }
    
    // Parallel reduction within the block to sum up partial sums from all threads
    __shared__ float sdata[1024]; // Max 1024 threads per block
    sdata[tid] = local_sum;
    __syncthreads();
    
    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Thread 0 writes the final sum for this batch item
    if (tid == 0) {
        out[batch_idx] = local_sum * scale_factor;
    }
}

torch::Tensor fused_gemm_sum_scale_cuda(torch::Tensor x, torch::Tensor weight, float scaling_factor) {
    auto batch_size = x.size(0);
    auto input_size = x.size(1);
    auto hidden_size = weight.size(0);
    
    // Output shape: [batch_size, 1]
    auto out = torch::zeros({batch_size, 1}, x.options());
    
    const int block_size = 1024;
    const int num_blocks = batch_size;
    
    float scale_val = scaling_factor / 2.0f; // The original code divides by 2 then multiplies by scaling_factor
    
    fused_gemm_sum_scale_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        input_size,
        hidden_size,
        scale_val
    );
    
    return out;
}
"""

custom_cpp_source = (
    "torch::Tensor fused_gemm_sum_scale_cuda(torch::Tensor x, torch::Tensor weight, float scaling_factor);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=custom_cpp_source,
    cuda_sources=custom_cuda_source,
    functions=["fused_gemm_sum_scale_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a matrix multiplication, division, summation, and scaling
    using a custom fused CUDA operator.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        # Use the fused CUDA operator
        return fused_ops.fused_gemm_sum_scale_cuda(x, self.weight, self.scaling_factor)