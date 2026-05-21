import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel source
# This kernel fuses: GEMM -> Max Reduction (dim 1) -> Mean Subtraction -> GELU
# It processes one batch element at a time to minimize global memory traffic and maximize register usage.
custom_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Helper for GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
__device__ inline float gelu(float x) {
    float cdf = 0.5f * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
    return x * cdf;
}

__global__ void fused_gemm_max_mean_gelu_kernel(
    const float* __restrict__ W, // Weight matrix: [out_features, in_features]
    const float* __restrict__ B, // Bias vector: [out_features]
    const float* __restrict__ X, // Input matrix: [batch_size, in_features]
    float* __restrict__ out,     // Output tensor: [batch_size, 1] (max values)
    int batch_size,
    int in_features,
    int out_features
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    const float* x_row = X + batch_idx * in_features;
    
    // We need to compute:
    // 1. y_j = sum_i(x_i * W_ij) + B_j for all j in [0, out_features)
    // 2. max_val = max_j(y_j)
    // 3. mean_val = (1/out_features) * sum_j(y_j)
    // 4. final_out = gelu(max_val - mean_val)

    // To avoid storing the entire intermediate vector y if out_features is large,
    // we can accumulate max and sum on-the-fly during the GEMM row computation.
    // However, standard GEMM computes dot products. Here we compute a full row of Y.
    
    // Since out_features (8192) fits in registers, we can store the intermediate results.
    // But to be safe and generic, let's use shared memory or just registers if small.
    // Given 8192 floats = 32KB, which is too large for registers per thread but fine for local memory/shared.
    // Let's use a simpler approach: Compute Y in registers/local memory, then reduce.
    
    // Actually, for out_features=8192, we can't hold all Y in registers easily without bank conflicts or spilling.
    // A better strategy for this specific fusion:
    // 1. Load a tile of X and W into shared memory? No, that's complex for GEMM + Reduction.
    // 2. Let's compute the full vector Y for the current batch item in local memory (array on stack).
    
    __shared__ float s_y[8192]; // Max out_features supported by this kernel config
    
    // Initialize shared memory or just use local array if we don't share between threads.
    // Each thread block handles one batch item. All threads in the block work together to compute Y.
    
    // We will have each thread compute a subset of the output features.
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    // Initialize max and sum for this batch item
    float local_max = -FLT_MAX;
    float local_sum = 0.0f;
    
    // We need to compute Y_j for all j.
    // Let's distribute the computation of Y across threads in the block.
    // Each thread computes a few Y_j values.
    
    // To do this efficiently, we can load X into shared memory once per block.
    __shared__ float s_x[8192]; // Max in_features
    
    // Load X row into shared memory
    for (int i = tid; i < in_features; i += num_threads) {
        s_x[i] = x_row[i];
    }
    __syncthreads();

    // Compute Y_j = dot(X, W_col_j) + B_j
    // We iterate over j. Each thread handles a subset of j.
    for (int j = tid; j < out_features; j += num_threads) {
        float sum = 0.0f;
        const float* w_col = W + j * in_features; // Column-major access is bad for GEMM usually, but here W is [out, in]
        // Wait, nn.Linear stores weight as [out_features, in_features]. 
        // So W[j][i] is at index j * in_features + i.
        
        #pragma unroll
        for (int i = 0; i < in_features; ++i) {
            sum += s_x[i] * w_col[i];
        }
        sum += B[j];
        
        // Store to shared memory if we want to reduce later, or just accumulate directly
        s_y[j] = sum;
    }
    __syncthreads();

    // Now reduce over the block to find max and sum of Y
    // We can use a parallel reduction in shared memory
    for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float val_y = s_y[tid];
            float val_y_next = s_y[tid + stride];
            
            // Update max
            if (val_y_next > val_y) val_y = val_y_next;
            s_y[tid] = val_y;
            
            // Update sum
            s_y[tid + num_threads] += s_y[tid + stride + num_threads]; 
            // Wait, we need separate storage for sum and max or use two arrays.
            // Let's use a second shared array for sum to avoid complexity.
        }
    }
    
    // The above reduction logic was flawed because I mixed max and sum in one array.
    // Let's restart the reduction part with two arrays: s_max and s_sum.
}

// Corrected Kernel with separate max and sum accumulation
__global__ void fused_gemm_max_mean_gelu_kernel_corrected(
    const float* __restrict__ W, 
    const float* __restrict__ B, 
    const float* __restrict__ X, 
    float* __restrict__ out, 
    int batch_size,
    int in_features,
    int out_features
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    const float* x_row = X + batch_idx * in_features;
    
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    // Shared memory for input row and intermediate results
    __shared__ float s_x[8192]; 
    __shared__ float s_y[8192];
    __shared__ float s_max[8192];
    __shared__ float s_sum[8192];

    // Load X into shared memory
    for (int i = tid; i < in_features; i += num_threads) {
        s_x[i] = x_row[i];
    }
    __syncthreads();

    // Compute Y_j for all j, distributed among threads
    for (int j = tid; j < out_features; j += num_threads) {
        float sum = 0.0f;
        const float* w_col = W + j * in_features; 
        
        #pragma unroll
        for (int i = 0; i < in_features; ++i) {
            sum += s_x[i] * w_col[i];
        }
        sum += B[j];
        
        s_y[j] = sum;
    }
    __syncthreads();

    // Initialize max and sum arrays for reduction
    if (tid < out_features) {
        s_max[tid] = s_y[tid];
        s_sum[tid] = s_y[tid];
    } else {
        s_max[tid] = -FLT_MAX;
        s_sum[tid] = 0.0f;
    }
    __syncthreads();

    // Parallel reduction for Max and Sum
    for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float val_max_next = s_max[tid + stride];
            float val_sum_next = s_sum[tid + stride];
            
            if (val_max_next > s_max[tid]) {
                s_max[tid] = val_max_next;
            }
            s_sum[tid] += val_sum_next;
        }
    }
    __syncthreads();

    // Thread 0 computes the final max and mean, applies GELU, and writes output
    if (tid == 0) {
        float final_max = s_max[0];
        float final_sum = s_sum[0];
        float mean_val = final_sum / out_features;
        
        float diff = final_max - mean_val;
        float result = gelu(diff);
        
        out[batch_idx] = result;
    }
}

torch::Tensor fused_gemm_max_mean_gelu_cuda(
    torch::Tensor x, 
    torch::Tensor w, 
    torch::Tensor b
) {
    auto batch_size = x.size(0);
    auto in_features = x.size(1);
    auto out_features = w.size(0);

    auto out = torch::zeros({batch_size}, x.options());

    const int block_size = 256; // Or 512, depending on shared memory usage
    // Shared memory usage: s_x (8192*4) + s_y (8192*4) + s_max (8192*4) + s_sum (8192*4) = 128KB
    // This might exceed shared memory limits for some GPUs if block_size is large, but 128KB is standard.
    // However, we only need s_x, s_max, s_sum. s_y can be avoided by computing max/sum on the fly?
    // No, because we need to reduce over all j. We can compute max/sum in registers per thread and then reduce.
    
    // Let's optimize shared memory usage:
    // We don't need s_y if we accumulate max/sum directly into s_max/s_sum during the GEMM loop?
    // No, because each thread computes a subset of j. We need to store the partial results for reduction.
    // Actually, we can just use s_max and s_sum as the storage for Y values initially.
    
    // Re-evaluating shared memory:
    // s_x: 8192 * 4 = 32KB
    // s_max: 8192 * 4 = 32KB (stores Y values initially, then max)
    // s_sum: 8192 * 4 = 32KB (stores Y values initially, then sum)
    // Total: 96KB. This fits in 100GB+ architecture shared memory (164KB).
    
    const int num_blocks = batch_size;

    fused_gemm_max_mean_gelu_kernel_corrected<<<num_blocks, block_size>>>(
        w.data_ptr<float>(), 
        b.data_ptr<float>(), 
        x.data_ptr<float>(), 
        out.data_ptr<float>(), 
        batch_size,
        in_features,
        out_features
    );

    return out;
}
"""

custom_cpp_source = (
    "torch::Tensor fused_gemm_max_mean_gelu_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=custom_cpp_source,
    cuda_sources=custom_cuda_source,
    functions=["fused_gemm_max_mean_gelu_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"]
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a fused GEMM, Max Reduction, Mean Subtraction, and GELU activation.
    """
    def __init__(self, in_features, out_features, max_dim):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_dim = max_dim
        # Weights and biases are stored as parameters but accessed directly in the kernel
        # Note: In a real scenario, you might want to register these as buffers or handle them differently
        # to ensure they are moved to GPU correctly. Here we assume they are passed from get_init_inputs logic
        # or handled by the caller. For this specific inline example, we will store them as buffers.
        self.register_buffer('weight', torch.empty(out_features, in_features))
        self.register_buffer('bias', torch.empty(out_features))

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor of shape (batch_size,) containing the GELU(max - mean) value for each batch item.
        """
        # The custom kernel expects weights and biases as separate tensors
        return fused_ops.fused_gemm_max_mean_gelu_cuda(x, self.weight, self.bias)

def get_inputs():
    return [torch.rand(1024, 8192)]

def get_init_inputs():
    in_features = 8192
    out_features = 8192
    max_dim = 1
    
    # Initialize the model to set up buffers
    model = ModelNew(in_features, out_features, max_dim)
    
    # Initialize weights and biases with random values similar to nn.Linear default initialization
    torch.nn.init.kaiming_uniform_(model.weight, a=math.sqrt(5))
    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(model.weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    torch.nn.init.uniform_(model.bias, -bound, bound)
    
    return [in_features, out_features, max_dim]

import math