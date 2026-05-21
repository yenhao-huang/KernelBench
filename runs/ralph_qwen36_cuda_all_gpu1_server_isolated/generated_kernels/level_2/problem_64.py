import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation: Gemm + LogSumExp + LeakyReLU + GELU
# We fuse these operations to reduce memory bandwidth pressure and improve performance.
# The input is (batch, in_features), weights are (out_features, in_features).
# Output is (batch, 1) after LogSumExp, then activations.

fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Helper for GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
__device__ __forceinline__ float gelu(float x) {
    return x * 0.5f * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
}

// Helper for LeakyReLU: max(0, x) + negative_slope * min(0, x)
__device__ __forceinline__ float leaky_relu(float x, float slope) {
    return fmaxf(0.0f, x) + slope * fminf(0.0f, x);
}

__global__ void fused_gemm_lse_leakyrelu_gelu_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, // Can be null if no bias
    float* __restrict__ output, 
    int batch_size, 
    int in_features, 
    int out_features, 
    float negative_slope) 
{
    int batch_idx = blockIdx.y;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || out_idx >= out_features) {
        return;
    }

    // Calculate the dot product for one output neuron
    float sum = 0.0f;
    const float* input_row = input + batch_idx * in_features;
    const float* weight_col = weight + out_idx * in_features; // Column-major access pattern optimization might be better, but let's stick to standard row-major for simplicity unless we change layout. 
    // Actually, PyTorch Linear uses (out, in) weights. input is (batch, in).
    // To optimize memory coalescing, we should iterate over in_features.
    
    #pragma unroll
    for (int i = 0; i < in_features; ++i) {
        sum += input_row[i] * weight_col[i];
    }

    if (bias != nullptr) {
        sum += bias[out_idx];
    }

    // Store intermediate result to shared memory or register? 
    // Since we need LogSumExp over the out_features dimension, we can't do it per-thread easily without reduction.
    // However, doing a full reduction inside the kernel for every thread is expensive.
    // Alternative: Use a two-pass approach or atomic adds if we were doing parallel reduction.
    // But wait, LogSumExp requires summing exp(x) over all out_features for a given batch.
    
    // Let's change strategy: 
    // 1. Compute Gemm result into a temporary buffer (or shared memory if small enough).
    // 2. Perform LogSumExp reduction per batch.
    // 3. Apply activations.
    
    // Given the constraints of inline CUDA and simplicity, let's assume we write the Gemm results to global memory first? 
    // No, that defeats the purpose of fusion if we read/write twice.
    
    // Better approach for LogSumExp fusion:
    // We can compute the Gemm result in registers, but we need to reduce across 'out_features'.
    // This suggests a block-level reduction.
    
    // Let's restructure the kernel launch configuration:
    // Each block handles one batch item? No, out_features is large (8192).
    // Each thread computes one output element? Then we need to reduce across threads in the block for LSE.
    
    // Let's use a grid-stride loop or specific block mapping.
    // Block size = 256. Grid x = ceil(out_features / 256). Grid y = batch_size.
    // Each thread computes one element of the Gemm output.
    // Then we need to reduce exp(val) across all threads in the same block (same batch, different out_idx).
    
    // This requires shared memory reduction for exp(sum).
}

// Since inline CUDA with complex reductions is verbose and error-prone without proper shared memory management,
// let's implement a simpler but still fused kernel that avoids intermediate global memory writes for the Gemm output.
// We will compute Gemm, then perform a parallel reduction within the block for LogSumExp.

__global__ void fused_gemm_lse_leakyrelu_gelu_optimized_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ output, 
    int batch_size, 
    int in_features, 
    int out_features, 
    float negative_slope) 
{
    extern __shared__ float shared_mem[];
    
    int batch_idx = blockIdx.y;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    // Each block processes one batch item.
    // Threads within the block compute different output features (out_features).
    // We need to reduce exp(Gemm_result) across all threads in the block.
    
    if (batch_idx >= batch_size) return;

    const float* input_row = input + batch_idx * in_features;
    
    // Shared memory for storing exp values and partial sums
    // Size needed: num_threads for exp values, plus some for reduction tree
    float* exp_vals = shared_mem;
    float* partial_sums = shared_mem + num_threads;

    // Step 1: Compute Gemm result for this thread's assigned output feature(s)
    // We use a grid-stride loop if out_features > num_threads, but let's assume out_features <= num_threads * blocks_x?
    // Actually, we launched Grid X = ceil(out_features / num_threads).
    // So each thread might handle multiple out_features if out_features is very large.
    
    float local_sum = 0.0f;
    int global_out_idx = blockIdx.x * num_threads + tid;
    
    for (int o = global_out_idx; o < out_features; o += num_threads * gridDim.x) {
        float val = 0.0f;
        const float* weight_col = weight + o * in_features;
        
        #pragma unroll
        for (int i = 0; i < in_features; ++i) {
            val += input_row[i] * weight_col[i];
        }
        
        if (bias != nullptr) {
            val += bias[o];
        }
        
        // Apply LeakyReLU and GELU sequentially as per the model definition
        // Note: The model applies LSE on the raw Gemm output, THEN activations.
        // So we must store the raw Gemm output (or at least enough info) to compute LSE first.
        // This breaks the "compute once" fusion if we apply activations before LSE.
        
        // Correction: The model is:
        // x = Linear(x)
        // x = LogSumExp(x, dim=1) -> Shape (batch, 1)
        // x = LeakyReLU(x)
        // x = LeakyReLU(x)
        // x = GELU(x)
        // x = GELU(x)
        
        // Since LSE reduces the dimension from (batch, out_features) to (batch, 1), 
        // the activations are applied to a scalar per batch item.
        // Therefore, we CANNOT fuse the activations into the Gemm kernel easily because they happen AFTER the reduction.
        
        // However, we CAN fuse: Gemm + LogSumExp.
        // The output of this fused op is (batch, 1).
        // Then we can apply LeakyReLU, LeakyReLU, GELU, GELU on that scalar.
        
        // So the kernel should compute:
        // For each batch b:
        //   max_val = max_j(Gemm(b, j))
        //   sum_exp = sum_j(exp(Gemm(b, j) - max_val))
        //   lse = max_val + log(sum_exp)
        //   output[b] = gelu(gelu(leaky_relu(leaky_relu(lse))))
        
        // This requires a parallel reduction within the block.
    }
    
    // Since we need to reduce across all out_features for each batch, and out_features can be large (8192),
    // we should map one block per batch item if possible, or use multiple blocks per batch with atomic adds.
    // Given 8192 out features, a single block of 256 threads is not enough to cover all features in one go without loops.
    
    // Let's restart the kernel design:
    // Block size: 256.
    // Grid Y: batch_size (1024).
    // Grid X: ceil(out_features / 256) = 32.
    // Each thread computes one element of the Gemm output.
    // We need to reduce exp(val) across all threads that belong to the same batch (same blockIdx.y).
    
    // Shared memory layout for reduction:
    // We'll use a standard parallel reduction for sum(exp(val)).
    // To handle numerical stability, we also need max(val).
    
    // Let's store val in shared memory first? No, too much memory.
    // We can compute max and sum_exp in registers per thread if we loop, but reduction needs shared mem.
    
    // Simplified approach: 
    // 1. Each thread computes its Gemm result `val`.
    // 2. Thread stores `exp(val)` into shared memory `exp_vals[tid]`.
    // 3. Perform parallel reduction on `exp_vals` to get `sum_exp`.
    // 4. Also need `max_val`. Store `val` in another shared array or compute max during reduction?
    //    Standard LSE: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x)))).
    //    So we need max(x) and sum(exp(x - max(x))).
    
    // Let's store `val` in shared memory.
    float* vals = shared_mem;
    vals[tid] = 0.0f; // Placeholder, will be filled below
    
    // Compute Gemm result
    float val = 0.0f;
    for (int o = global_out_idx; o < out_features; o += num_threads * gridDim.x) {
        float v = 0.0f;
        const float* weight_col = weight + o * in_features;
        #pragma unroll
        for (int i = 0; i < in_features; ++i) {
            v += input_row[i] * weight_col[i];
        }
        if (bias != nullptr) {
            v += bias[o];
        }
        // If multiple outputs per thread, we need to handle LSE over them too? 
        // No, the reduction is across ALL out_features.
        // If a thread handles multiple features, it must reduce them locally first or store all.
        // Storing all is expensive.
        // Let's assume out_features <= num_threads * gridDim.x and each thread handles exactly one feature for simplicity?
        // 8192 / 256 = 32 blocks. 32 * 256 = 8192. Perfect fit.
        if (global_out_idx < out_features) {
             val = v;
             break; // Only one feature per thread in this configuration
        }
    }
    
    // If global_out_idx >= out_features, val is undefined/zero, which is fine as it won't be used.
    if (global_out_idx < out_features) {
        vals[tid] = val;
    } else {
        vals[tid] = -INFINITY;
    }
    
    __syncthreads();
    
    // Parallel reduction for max and sum_exp
    // We need to compute: max_val = max(vals), sum_exp = sum(exp(vals - max_val))
    
    // Step 1: Find max_val in shared memory
    for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            vals[tid] = fmaxf(vals[tid], vals[tid + stride]);
        }
        __syncthreads();
    }
    
    float max_val = vals[0];
    
    // Step 2: Compute sum(exp(val - max_val))
    // We can reuse the shared memory array or use a separate one. Let's use a second part of shared mem.
    float* exp_vals = shared_mem + num_threads;
    
    if (global_out_idx < out_features) {
        exp_vals[tid] = expf(vals[tid] - max_val);
    } else {
        exp_vals[tid] = 0.0f;
    }
    
    __syncthreads();
    
    // Parallel reduction for sum
    for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            exp_vals[tid] += exp_vals[tid + stride];
        }
        __syncthreads();
    }
    
    float sum_exp = exp_vals[0];
    
    // Step 3: Compute LSE
    float lse = max_val + logf(sum_exp);
    
    // Step 4: Apply activations to the scalar LSE result
    // LeakyReLU(LeakyReLU(GELU(GELU(lse))))
    // Note: The order in the model is LSE -> LeakyReLU -> LeakyReLU -> GELU -> GELU
    
    lse = leaky_relu(leaky_relu(gelu(gelu(lse)), negative_slope), negative_slope);
    
    // Write output
    if (tid == 0) {
        output[batch_idx] = lse;
    }
}

torch::Tensor fused_gemm_lse_leakyrelu_gelu_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias) 
{
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    
    auto output = torch::empty({batch_size, 1}, input.options());
    
    const int block_size = 256;
    const int grid_x = (out_features + block_size - 1) / block_size;
    const int grid_y = batch_size;
    
    // Shared memory size: 2 * block_size floats
    const int shared_mem_size = 2 * block_size * sizeof(float);
    
    float negative_slope = 0.01f;
    
    fused_gemm_lse_leakyrelu_gelu_optimized_kernel<<<grid_x, grid_y, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.numel() > 0 ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        negative_slope
    );
    
    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_gemm_lse_leakyrelu_gelu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_gemm_lse_leakyrelu_gelu_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA operator for Gemm + LogSumExp + LeakyReLU + GELU fusion.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        
        # Initialize weights and biases manually to match nn.Linear behavior
        # nn.Linear uses Kaiming uniform initialization by default
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weight with Kaiming uniform
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Perform the fused operation
        # Input x: (batch, in_features)
        # Weight: (out_features, in_features)
        # Bias: (out_features,)
        
        # Ensure inputs are contiguous and on CUDA
        if not x.is_contiguous():
            x = x.contiguous()
        if not self.weight.is_contiguous():
            self.weight.data = self.weight.data.contiguous()
        if self.bias is not None and not self.bias.is_contiguous():
            self.bias.data = self.bias.data.contiguous()
            
        output = fused_ops.fused_gemm_lse_leakyrelu_gelu_cuda(x, self.weight, self.bias)
        
        return output

import math