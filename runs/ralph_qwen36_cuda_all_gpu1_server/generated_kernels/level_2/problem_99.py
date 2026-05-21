import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Linear + GELU + Softmax fusion
# This kernel performs: y = softmax(gelu(x @ W^T + b))
# Optimized for FP32 precision.
# We assume input x is (N, K), weight W is (M, K) (nn.Linear stores weights as (out_features, in_features)), bias b is (M).
# Output y is (N, M).

custom_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Helper for atomic max if needed, but here we use a two-pass approach or online softmax.
// For simplicity and robustness with large dimensions, we'll use a standard two-pass softmax 
// after computing the linear + gelu values. However, to be truly optimized, we can fuse the reduction.
// Given the constraints of inline code and complexity, a fused kernel that computes 
// val = gelu(linear(x)) and then performs softmax is ideal.

__device__ __forceinline__ float gelu_forward(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_linear_gelu_softmax_kernel(
    const float* __restrict__ x, 
    const float* __restrict__ w, 
    const float* __restrict__ b, 
    float* __restrict__ out, 
    int batch_size, 
    int in_features, 
    int out_features) 
{
    // Each block handles one row of the output (one sample in the batch)
    // However, for large out_features, we might want multiple blocks per row or grid-stride loops.
    // Let's assign one thread block to one output element? No, that's too many blocks.
    // Let's assign one thread block to one sample (row of x).
    
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    extern __shared__ float shared_mem[];
    
    // shared_mem[0..out_features-1] will store the computed values for this batch item before softmax reduction
    // shared_mem[out_features..2*out_features-1] can be used for intermediate sums if needed, 
    // but we'll do a two-pass approach within the block or use atomic operations.
    // Actually, standard efficient softmax uses two passes: max and sum.
    // With shared memory, we can compute the local max and sum for the block's portion of the row?
    // No, each block handles one full row (out_features). So we need to reduce across threads in the block.
    
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    // Each thread computes a subset of the output features for this batch item
    float local_max = -INFINITY;
    float local_sum = 0.0f;
    
    // We need to store the computed values in shared memory to perform the softmax reduction
    // Since out_features can be large (8192), we might not fit everything in shared memory if we have many threads.
    // But 8192 floats is 32KB, which fits in shared memory on most GPUs.
    
    float* vals = shared_mem; 
    
    // Compute linear + gelu for this batch item
    // x is (batch_size, in_features), w is (out_features, in_features)
    // We want out[batch_idx, j] = gelu(sum_k x[batch_idx, k] * w[j, k] + b[j])
    
    // To optimize memory access, we can load w[j] into registers or shared memory if small, 
    // but here we iterate over j (out_features) and k (in_features).
    // It's better to have each thread compute one output feature j.
    
    for (int j = tid; j < out_features; j += num_threads) {
        float sum = 0.0f;
        const float* x_row = x + batch_idx * in_features;
        const float* w_j = w + j * in_features;
        
        // Dot product
        for (int k = 0; k < in_features; ++k) {
            sum += x_row[k] * w_j[k];
        }
        
        // Add bias
        sum += b[j];
        
        // Apply GELU
        float gelu_val = gelu_forward(sum);
        
        vals[j] = gelu_val;
    }
    
    __syncthreads();
    
    // Now perform softmax reduction on vals[0...out_features-1]
    // Pass 1: Find max
    float block_max = -INFINITY;
    for (int j = tid; j < out_features; j += num_threads) {
        if (vals[j] > block_max) {
            block_max = vals[j];
        }
    }
    
    // Reduce max across threads in the block
    // Simple reduction using shared memory or just atomicMax? 
    // Since we are in one block, we can use a tree reduction.
    for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (tid < stride) {
            if (vals[tid] > vals[tid + stride]) { // This is wrong, we need to reduce the max variable
                // We need a separate shared array for reduction or use registers.
                // Let's use a simple approach: store max in shared memory and reduce.
            }
        }
    }
    
    // Better reduction for max:
    __shared__ float s_max;
    if (tid == 0) s_max = -INFINITY;
    __syncthreads();
    
    // Update global max with local max
    atomicMax_float(&s_max, block_max); // atomicMax doesn't exist for float
    
    // Let's use a simpler reduction loop for max
    // We'll just do a naive reduction in shared memory
    __shared__ float s_vals_max[1024]; // Assuming num_threads <= 1024
    if (tid < 1024) s_vals_max[tid] = block_max;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_vals_max[tid + stride] > s_vals_max[tid]) {
                s_vals_max[tid] = s_vals_max[tid + stride];
            }
        }
        __syncthreads();
    }
    
    float max_val = s_vals_max[0];
    
    // Pass 2: Compute sum of exp(x - max)
    float block_sum = 0.0f;
    for (int j = tid; j < out_features; j += num_threads) {
        vals[j] = expf(vals[j] - max_val);
        block_sum += vals[j];
    }
    
    // Reduce sum across threads
    __shared__ float s_vals_sum[1024];
    if (tid < 1024) s_vals_sum[tid] = block_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_vals_sum[tid] += s_vals_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float sum_val = s_vals_sum[0];
    
    // Pass 3: Normalize
    for (int j = tid; j < out_features; j += num_threads) {
        out[batch_idx * out_features + j] = vals[j] / sum_val;
    }
}

// Helper to get atomicMax for float
__device__ __forceinline__ float atomicMax_float(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
            __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (old != assumed);
    return __int_as_float(old);
}

torch::Tensor fused_linear_gelu_softmax_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
    auto batch_size = x.size(0);
    auto in_features = x.size(1);
    auto out_features = w.size(0);
    
    auto out = torch::zeros({batch_size, out_features}, x.options());
    
    const int block_size = 256; // Or 512, depending on shared memory usage
    // Shared memory size: out_features * sizeof(float) + reduction buffers
    // For out_features=8192, we need ~32KB for vals. 
    // We can launch one block per batch item if block_size is large enough to cover out_features?
    // No, 8192 threads per block is too many (max is usually 1024).
    // So each block handles one batch item, but threads iterate over out_features.
    
    const int num_blocks = batch_size;
    
    // Calculate shared memory size: 
    // vals array: out_features * 4 bytes
    // s_vals_max: min(block_size, 1024) * 4 bytes
    // s_vals_sum: min(block_size, 1024) * 4 bytes
    int shared_mem_size = (out_features + 2 * std::min(block_size, 1024)) * sizeof(float);
    
    fused_linear_gelu_softmax_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        x.data_ptr<float>(), 
        w.data_ptr<float>(), 
        b.data_ptr<float>(), 
        out.data_ptr<float>(), 
        batch_size, 
        in_features, 
        out_features
    );
    
    return out;
}
"""

custom_ops_cpp_source = (
    "torch::Tensor fused_linear_gelu_softmax_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=custom_ops_cpp_source,
    cuda_sources=custom_ops_source,
    functions=["fused_linear_gelu_softmax_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Optimized model using fused CUDA operator for Linear + GELU + Softmax.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        # Store the fused function reference
        self.fused_op = fused_ops.fused_linear_gelu_softmax_cuda

    def forward(self, x):
        w = self.linear.weight  # Shape: (out_features, in_features)
        b = self.linear.bias    # Shape: (out_features,)
        
        # Call the fused CUDA kernel
        out = self.fused_op(x, w, b)
        
        return out

# Helper functions to match the interface
def get_inputs():
    batch_size = 1024
    in_features = 8192
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    in_features = 8192
    out_features = 8192
    return [in_features, out_features]