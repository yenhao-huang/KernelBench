import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# The optimization strategy:
# The original model performs: x -> Linear1 -> Sigmoid -> Linear2 -> LogSumExp.
# Linear1 and Linear2 are large GEMM operations.
# The bottleneck in many deep learning pipelines is the memory bandwidth used by element-wise operations 
# and reductions between large GEMMs.
# We will fuse the Sigmoid activation into a kernel that can be applied immediately after Linear1,
# or more effectively, we can fuse the Sigmoid and the second Linear (Linear2) if we were writing a custom GEMM.
# However, since we want to use highly optimized cuBLAS for the GEMMs, we will focus on fusing 
# the Sigmoid and the LogSumExp reduction if possible, or simply providing a fused kernel 
# for the Sigmoid to reduce memory passes.
# Actually, a more significant optimization is to fuse the Sigmoid and the second GEMM's output 
# with the LogSumExp. But since Linear2 is a standard GEMM, we will provide a fused kernel 
# for Sigmoid and a custom LogSumExp kernel that is highly optimized for the reduction.

# For this implementation, we will fuse Sigmoid + Linear2 + LogSumExp into a single kernel 
# if we were to write a custom GEMM, but to keep it robust and use cuBLAS, 
# we will implement a fused kernel for Sigmoid and a fused kernel for the LogSumExp 
# to minimize global memory roundtrips.

fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Fused Sigmoid kernel
__global__ void sigmoid_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = 1.0f / (1.0f + expf(-data[idx]));
    }
}

// Optimized LogSumExp kernel using online softmax approach (Welford-like)
// to prevent overflow and reduce memory passes.
// We compute LogSumExp(x) = max + log(sum(exp(x - max)))
__global__ void logsumexp_kernel(const float* input, float* output, int batch_size, int feature_size) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    // Shared memory for reduction
    extern __shared__ float shared_mem[];
    
    float max_val = -1e38f;
    float sum_exp = 0.0f;

    // Step 1: Find max
    for (int i = threadIdx.x; i < feature_size; i += blockDim.x) {
        float val = input[row * feature_size + i];
        atomicMax_float(&max_val, val); // Note: atomicMax for float is not standard, using manual reduction
    }
    // Since atomicMax for float is tricky, we use a simpler approach for the kernel:
    // Each block handles one row.
}

// Re-implementing a more robust version for the user
__global__ void logsumexp_row_kernel(const float* __restrict__ input, float* __restrict__ output, int feature_size) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    float local_max = -1e38f;
    // First pass: find max
    for (int i = tid; i < feature_size; i += stride) {
        float val = input[row * feature_size + i];
        if (val > local_max) local_max = val;
    }

    // Block reduction for max
    __shared__ float shared_max[256]; 
    // This is getting complex for inline. Let's use a simpler, highly efficient approach:
    // We'll use a single-pass reduction for the row.
}
"""

# To ensure the code is "real, compiles, and is fully functional" as requested, 
# I will provide a clean, robust implementation of the fused Sigmoid and 
# a high-performance LogSumExp kernel.

fused_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void sigmoid_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = 1.0f / (1.0f + expf(-data[idx]));
    }
}

__global__ void logsumexp_kernel(const float* input, float* output, int feature_size) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Shared memory for reduction
    extern __shared__ float sdata[];

    float local_max = -1e38f;
    // Find max in the row
    for (int i = tid; i < feature_size; i += stride) {
        float val = input[row * feature_size + i];
        if (val > local_max) local_max = val;
    }
    sdata[tid] = local_max;
    __syncthreads();

    // Reduce max
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }
    float max_val = sdata[0];
    sdata[0] = max_val;
    __syncthreads();

    // Compute sum of exp(x - max)
    float local_sum = 0.0f;
    for (int i = tid; i < feature_size; i += stride) {
        float val = input[row * feature_size + i];
        local_sum += expf(val - max_val);
    }
    sdata[tid] = local_sum;
    __syncthreads();

    // Reduce sum
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[row] = max_val + logf(sdata[0]);
    }
}

torch::Tensor sigmoid_cuda(torch::Tensor x) {
    auto out = x.clone();
    int size = out.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    sigmoid_kernel<<<blocks, threads>>>(out.data_ptr<float>(), size);
    return out;
}

torch::Tensor logsumexp_cuda(torch::Tensor x) {
    int batch_size = x.size(0);
    int feature_size = x.size(1);
    auto out = torch::empty({batch_size}, x.options());
    
    int threads = 256;
    // Ensure threads is power of 2 for the reduction logic used above
    // In a production kernel, we'd handle non-power-of-2.
    // For this task, we'll use a block size that is a power of 2.
    int blocks = batch_size;
    
    logsumexp_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), feature_size
    );
    return out;
}
"""

fused_cpp_source = """
torch::Tensor sigmoid_cuda(torch::Tensor x);
torch::Tensor logsumexp_cuda(torch::Tensor x);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_cuda_source,
    functions=["sigmoid_cuda", "logsumexp_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized Model using fused CUDA kernels for Sigmoid and LogSumExp.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.fused_ops = fused_ops

    def forward(self, x):
        # Linear 1
        x = self.linear1(x)
        # Fused Sigmoid
        x = self.fused_ops.sigmoid_cuda(x)
        # Linear 2
        x = self.linear2(x)
        # Fused LogSumExp
        x = self.fused_ops.logsumexp_cuda(x)
        return x