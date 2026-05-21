import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused matmul + max_pool1d + sum + scale
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_matmul_maxpool_sum_scale_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_features, int out_features,
    int kernel_size, float scale_factor) {
    
    // Each block handles one batch element
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    // Shared memory for weight matrix (tiled)
    extern __shared__ float shared_weight[];
    
    // Thread index within block
    int tid = threadIdx.x;
    int out_idx = tid;
    
    // Initialize accumulator for this output feature
    float acc = 0.0f;
    
    // Process input in tiles
    for (int tile_start = 0; tile_start < in_features; tile_start += blockDim.x) {
        int in_idx = tile_start + tid;
        
        // Load weight tile into shared memory
        if (in_idx < in_features && out_idx < out_features) {
            shared_weight[tid] = weight[out_idx * in_features + in_idx];
        } else {
            shared_weight[tid] = 0.0f;
        }
        __syncthreads();
        
        // Compute partial dot product
        if (out_idx < out_features) {
            float partial = 0.0f;
            for (int k = 0; k < blockDim.x; k++) {
                int actual_in = tile_start + k;
                if (actual_in < in_features) {
                    partial += input[batch_idx * in_features + actual_in] * shared_weight[k];
                }
            }
            acc += partial;
        }
        __syncthreads();
    }
    
    // Add bias
    if (out_idx < out_features) {
        acc += bias[out_idx];
    }
    
    // Now we have the linear output for this batch element
    // Apply max_pool1d with kernel_size
    // We need to compute max over sliding windows of size kernel_size
    // For simplicity, we assume stride = kernel_size (default for MaxPool1d)
    // Actually, PyTorch default stride is kernel_size, so output length = out_features / kernel_size
    
    // We'll compute max pool in shared memory
    // First, store linear output in shared memory
    extern __shared__ float shared_data[];
    float* linear_out = shared_data;
    float* pool_out = &shared_data[out_features];
    
    if (out_idx < out_features) {
        linear_out[out_idx] = acc;
    }
    __syncthreads();
    
    // Compute max pool
    int pool_out_size = out_features / kernel_size;
    if (out_idx < pool_out_size) {
        float max_val = -1e30f;
        for (int k = 0; k < kernel_size; k++) {
            int idx = out_idx * kernel_size + k;
            if (idx < out_features) {
                max_val = fmaxf(max_val, linear_out[idx]);
            }
        }
        pool_out[out_idx] = max_val;
    }
    __syncthreads();
    
    // Compute sum over pool output
    // Use parallel reduction
    if (out_idx < pool_out_size) {
        // Simple sum reduction using shared memory
        extern __shared__ float shared_sum[];
        shared_sum[out_idx] = pool_out[out_idx];
        __syncthreads();
        
        // Reduction
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (out_idx < stride && out_idx + stride < pool_out_size) {
                shared_sum[out_idx] += shared_sum[out_idx + stride];
            }
            __syncthreads();
        }
        
        // Thread 0 writes the scaled sum
        if (out_idx == 0) {
            output[batch_idx] = shared_sum[0] * scale_factor;
        }
    }
}

torch::Tensor fused_matmul_maxpool_sum_scale_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, float scale_factor) {
    
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::empty({batch_size}, input.options());
    
    const int threads = 256;  // Must be power of 2 for reduction
    const int blocks = batch_size;
    
    // Calculate shared memory size
    // We need: weight tile (threads floats), linear_out (out_features floats), 
    // pool_out (out_features/kernel_size floats), reduction (out_features/kernel_size floats)
    int pool_out_size = out_features / kernel_size;
    int shared_mem_size = (threads + out_features + pool_out_size + pool_out_size) * sizeof(float);
    
    fused_matmul_maxpool_sum_scale_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, in_features, out_features,
        kernel_size, scale_factor);
    
    return output;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_matmul_maxpool_sum_scale_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, float scale_factor);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_matmul_maxpool_sum_scale",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_matmul_maxpool_sum_scale_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor
        self.fused_op = fused_op

    def forward(self, x):
        # Use the fused CUDA operator that combines matmul, max_pool, sum, and scale
        return self.fused_op.fused_matmul_maxpool_sum_scale_cuda(
            x, self.matmul.weight, self.matmul.bias,
            self.kernel_size, self.scale_factor
        )