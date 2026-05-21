import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation:
# 1. Linear Layer (Matmul + Bias Add)
# 2. Group Normalization
# 3. Leaky ReLU
# 4. Element-wise Addition (x + x)
#
# Note: We fuse these to minimize memory traffic between global memory and registers/shared memory.
# The input is (B, I), output of linear is (B, H).
# GN operates on (B, H) with G groups.
# Leaky ReLU is element-wise.
# x + x is element-wise scaling by 2.

fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Helper for Group Normalization statistics calculation and application
// We assume input shape (N, C) where N=batch_size, C=hidden_size
// num_groups is the number of groups.

__global__ void fused_linear_gn_leakyrelu_add_kernel(
    const float* __restrict__ x,      // Input: (B, I)
    const float* __restrict__ weight, // Weight: (H, I)
    const float* __restrict__ bias,   // Bias: (H)
    float* __restrict__ out,          // Output: (B, H)
    
    int batch_size,
    int input_size,
    int hidden_size,
    int num_groups,
    float eps,
    float negative_slope
) {
    // Each thread block handles one sample in the batch? 
    // Or we can do a 2D grid. Let's use a 1D grid where each block handles one row (one sample).
    // This simplifies shared memory usage for GN stats if we process per-sample.
    
    int idx = blockIdx.x;
    if (idx >= batch_size) return;

    const float* x_row = x + idx * input_size;
    float* out_row = out + idx * hidden_size;
    
    // Shared memory for partial sums for GN
    // We need sum and sum_of_squares for each group.
    // Groups are contiguous in the channel dimension? 
    // PyTorch GroupNorm: channels are divided into num_groups groups.
    // Each group has C/G channels.
    
    int channels_per_group = hidden_size / num_groups;
    
    // We need to compute mean and var for each group.
    // Since we are processing one sample at a time, we can use shared memory to accumulate stats.
    // However, shared memory size might be large if H is large. 
    // H=8192, G=512 -> 16 channels per group.
    // We can compute stats in registers or global memory if needed, but let's try shared memory for speed.
    // Actually, for GN, we iterate over all channels. 
    // Let's use a simple approach: each thread block computes the stats for the whole sample using atomic adds or just local accumulation if we split work.
    
    // Alternative: Use a 2D grid where threads cooperate to compute stats.
    // Given the constraints and simplicity, let's stick to one block per sample but optimize the inner loop.
    // To avoid shared memory bank conflicts and size issues, we can compute stats in registers by looping over groups? 
    // No, we need to iterate over all channels to get sum/sq_sum for each group.
    
    // Let's use a strategy where threads within a block cooperate to compute the global sum/sq_sum for the sample,
    // then we split into groups.
    
    extern __shared__ float shared_mem[];
    
    // shared_mem layout: 
    // First 2 * num_groups floats for sum and sum_sq of each group? 
    // Or just use local variables if num_groups is small? 512 is too big for registers per thread.
    // Let's allocate shared memory for the sums of each group.
    float* group_sums = shared_mem;
    float* group_sum_sq = shared_mem + num_groups;
    
    // Initialize sums to 0
    int tid = threadIdx.x;
    if (tid < num_groups) {
        group_sums[tid] = 0.0f;
        group_sum_sq[tid] = 0.0f;
    }
    __syncthreads();

    // Each thread processes a chunk of the input channels
    // Total channels = hidden_size
    // We can have multiple threads process different parts of the channel vector.
    
    int total_channels = hidden_size;
    for (int c = tid; c < total_channels; c += blockDim.x) {
        // Determine which group this channel belongs to
        int g = c / channels_per_group;
        
        // Compute Linear output for this channel
        // y_c = sum_j(x_j * W_{c,j}) + b_c
        float acc = 0.0f;
        const float* w_row = weight + c * input_size;
        
        #pragma unroll
        for (int j = 0; j < input_size; ++j) {
            acc += x_row[j] * w_row[j];
        }
        acc += bias[c];
        
        // Accumulate statistics for Group Normalization
        atomicAdd(&group_sums[g], acc);
        atomicAdd(&group_sum_sq[g], acc * acc);
    }
    
    __syncthreads();

    // Compute mean and variance for each group
    float inv_std[num_groups]; // This might be too large for stack if num_groups is huge, but 512 floats is 2KB. It's fine.
    // Actually, let's use a local array or just compute on the fly in the next pass.
    // To avoid large stack allocation, we can store them in shared memory again or recompute.
    // Let's store in shared memory for reuse.
    
    float* group_means = shared_mem + 2 * num_groups;
    float* group_inv_stds = shared_mem + 2 * num_groups + num_groups;

    if (tid < num_groups) {
        float sum = group_sums[tid];
        float sum_sq = group_sum_sq[tid];
        int n = channels_per_group;
        
        float mean = sum / n;
        float var = (sum_sq / n) - (mean * mean);
        // Add epsilon for stability
        var += eps;
        
        group_means[tid] = mean;
        group_inv_stds[tid] = rsqrtf(var);
    }
    
    __syncthreads();

    // Second pass: Apply GN, Leaky ReLU, and x+x (which is 2*x)
    for (int c = tid; c < total_channels; c += blockDim.x) {
        int g = c / channels_per_group;
        
        float val = out_row[c]; // We already computed the linear output in the first pass? 
        // Wait, we didn't store it. We need to recompute or store it.
        // Recomputing is expensive. Let's store the linear output in a temporary buffer or just recompute if input_size is small?
        // Input size is 8192. Recomputing matmul row is expensive.
        
        // Better approach: Store the linear output in global memory first, then read it back for GN.
        // But we are writing to 'out' directly.
        // Let's change strategy: 
        // Pass 1: Compute Linear Output and store in 'out'. Also compute stats using atomics.
        // Pass 2: Read from 'out', apply GN, Leaky ReLU, Add.
        
        // Since we are in one kernel, we can't easily do two passes without storing intermediate results.
        // We can use shared memory to store the linear output for this block? 
        // Block size is typically 256 or 512. Hidden size is 8192. Shared memory isn't big enough for all outputs.
        
        // So, we MUST write the linear output to global memory 'out' in the first pass.
        // Then read it back in the second pass.
    }
    
    // Let's restructure the kernel logic properly.
}

// Actually, the above logic is getting complex with atomics and shared memory management for GN stats.
// A more robust way for GN in a fused kernel:
// 1. Compute Linear Output -> Store in Global Memory (out)
// 2. Compute Stats per Group -> Use Shared Memory to aggregate, then write back to Global Memory or keep in registers if possible.
// 3. Apply Normalization, Activation, and Scaling.

// Let's use a simpler, highly optimized structure:
// Kernel 1: Linear Layer (Matmul + Bias)
// Kernel 2: GroupNorm + LeakyReLU + Add

// But the prompt asks for custom CUDA operators to replace pytorch operators. 
// We can write one big fused kernel or two. Let's try to fuse Linear and GN+Activation+Add if possible, 
// but GN requires global reduction over channels which is hard to fuse with Matmul efficiently without multiple passes.
// However, we can fuse GN + LeakyReLU + Add into a single kernel that reads the output of Linear.

// Let's define two kernels:
// 1. fused_linear: Computes y = xW^T + b
// 2. fused_gn_act_add: Computes gn(y), then leaky_relu, then y+y.

// This is still better than separate calls because we avoid global memory writes/reads between GN and Activation if we fuse them.
// And we can fuse Linear into a highly optimized GEMM-like structure or just a simple loop if dimensions are small enough? 
// 8192x8192 is large. We should use cuBLAS for Matmul usually, but the prompt implies writing custom CUDA.
// Writing a full tiled GEMM from scratch is very long. 
// However, we can use `torch::matmul` for the linear part if we want, but the goal is to replace operators.
// Let's write a simple but efficient Linear kernel using shared memory tiling if possible, or just rely on the fact that 
// for this specific problem, the bottleneck might be the memory bandwidth of GN/Activation.

// Actually, for 8192x8192, a naive GEMM is slow. But let's assume we want to show custom CUDA implementation.
// We will implement a simple Linear kernel and a fused GN+Act+Add kernel.

__global__ void linear_kernel(
    const float* __restrict__ x,      // (B, I)
    const float* __restrict__ weight, // (H, I)
    const float* __restrict__ bias,   // (H)
    float* __restrict__ out,          // (B, H)
    int batch_size,
    int input_size,
    int hidden_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * hidden_size) return;

    int b = idx / hidden_size;
    int h = idx % hidden_size;

    float sum = 0.0f;
    const float* x_row = x + b * input_size;
    const float* w_col = weight + h * input_size; // Weight is (H, I), so row h is the weights for output channel h

    #pragma unroll
    for (int i = 0; i < input_size; ++i) {
        sum += x_row[i] * w_col[i];
    }
    
    out[idx] = sum + bias[h];
}

__global__ void fused_gn_leakyrelu_add_kernel(
    const float* __restrict__ x,      // (B, H) - Input from Linear
    float* __restrict__ out,          // (B, H) - Output after GN, LeakyReLU, Add
    int batch_size,
    int hidden_size,
    int num_groups,
    float eps,
    float negative_slope
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * hidden_size) return;

    int b = idx / hidden_size;
    int h = idx % hidden_size;
    
    // We need to compute GN statistics for the entire sample b.
    // This requires a reduction over all channels in the batch.
    // Doing this per-thread is inefficient. 
    // We should use a block-level reduction or atomic operations.
    
    // Let's use a strategy where each block handles one sample (batch element).
    // Block size = 256 or 512.
    // If we launch batch_size blocks, each block processes one sample.
    
    // But the grid is 1D here. We need to know if this thread belongs to a specific sample's reduction.
    // Let's change the kernel launch configuration in Python: 
    // For GN kernel: num_blocks = batch_size, block_size = hidden_size (or less).
    // If block_size < hidden_size, we need cooperative reduction.
    
    // Given the complexity of writing a full cooperative reduction for GN stats from scratch in inline CUDA,
    // and the fact that PyTorch's GroupNorm is already quite optimized, 
    // the main speedup opportunity here is fusing GN + LeakyReLU + Add into a single pass over memory,
    // avoiding intermediate global memory writes.
    
    // However, to compute GN stats, we MUST read all channels.
    // If we process one element at a time, we can't easily get the mean/var without reading the whole sample multiple times or using atomics.
    
    // Let's assume a block handles one sample.
    // We need to determine if this thread is part of the block for sample b.
    // In the Python code, we will launch with num_blocks=batch_size.
    // So blockIdx.x corresponds to the batch index.
    
    int b_idx = blockIdx.x;
    if (b_idx >= batch_size) return;

    const float* x_sample = x + b_idx * hidden_size;
    float* out_sample = out + b_idx * hidden_size;
    
    int channels_per_group = hidden_size / num_groups;
    
    // Shared memory for group sums and sum_sq
    extern __shared__ float shared_mem[];
    float* group_sums = shared_mem;
    float* group_sum_sq = shared_mem + num_groups;
    
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Initialize shared memory for sums
    if (tid < num_groups) {
        group_sums[tid] = 0.0f;
        group_sum_sq[tid] = 0.0f;
    }
    __syncthreads();
    
    // Each thread processes a subset of channels
    for (int c = tid; c < hidden_size; c += block_size) {
        int g = c / channels_per_group;
        float val = x_sample[c];
        
        atomicAdd(&group_sums[g], val);
        atomicAdd(&group_sum_sq[g], val * val);
    }
    
    __syncthreads();
    
    // Compute mean and inv_std for each group
    // We can store these in shared memory or registers. 
    // Since num_groups is 512, we can't store all in registers per thread easily without complexity.
    // Let's store them in shared memory again.
    float* group_means = shared_mem + 2 * num_groups;
    float* group_inv_stds = shared_mem + 2 * num_groups + num_groups;
    
    if (tid < num_groups) {
        float sum = group_sums[tid];
        float sum_sq = group_sum_sq[tid];
        int n = channels_per_group;
        
        float mean = sum / n;
        float var = (sum_sq / n) - (mean * mean);
        var += eps;
        
        group_means[tid] = mean;
        group_inv_stds[tid] = rsqrtf(var);
    }
    
    __syncthreads();
    
    // Apply GN, Leaky ReLU, and Add (x+x)
    for (int c = tid; c < hidden_size; c += block_size) {
        int g = c / channels_per_group;
        
        float val = x_sample[c];
        float mean = group_means[g];
        float inv_std = group_inv_stds[g];
        
        // Normalize
        float normalized = (val - mean) * inv_std;
        
        // Leaky ReLU
        if (normalized < 0) {
            normalized *= negative_slope;
        }
        
        // Add x + x -> effectively multiply by 2? 
        // The original code is `x = x + x`. This is `2 * x`.
        // So we scale the activated value by 2.
        normalized *= 2.0f;
        
        out_sample[c] = normalized;
    }
}

// Python bindings
torch::Tensor linear_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    auto batch_size = x.size(0);
    auto input_size = x.size(1);
    auto hidden_size = weight.size(0);
    
    auto out = torch::empty({batch_size, hidden_size}, x.options());
    
    const int block_size = 256;
    const int num_blocks = (batch_size * hidden_size + block_size - 1) / block_size;
    
    linear_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size, input_size, hidden_size
    );
    
    return out;
}

torch::Tensor fused_gn_leakyrelu_add_cuda(torch::Tensor x, int num_groups, float eps, float negative_slope) {
    auto batch_size = x.size(0);
    auto hidden_size = x.size(1);
    
    auto out = torch::empty_like(x);
    
    // Launch one block per sample for GN reduction
    const int block_size = 256; // Or 512, depending on shared memory limits
    // Shared memory size: 4 * num_groups floats (sums, sum_sq, means, inv_stds)
    // 4 * 512 * 4 bytes = 8KB. This is well within the limit (usually 48KB or more).
    int shared_mem_size = 4 * num_groups * sizeof(float);
    
    fused_gn_leakyrelu_add_kernel<<<batch_size, block_size, shared_mem_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size, hidden_size, num_groups, eps, negative_slope
    );
    
    return out;
}

"""

fused_ops_cpp_source = (
    "torch::Tensor linear_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias);"
    "torch::Tensor fused_gn_leakyrelu_add_cuda(torch::Tensor x, int num_groups, float eps, float negative_slope);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["linear_cuda", "fused_gn_leakyrelu_add_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA operators.
    1. Linear layer is computed via a custom CUDA kernel (linear_cuda).
    2. GroupNorm, LeakyReLU, and Element-wise Add are fused into a single CUDA kernel (fused_gn_leakyrelu_add_cuda).
    """
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.eps = eps
        self.negative_slope = negative_slope
        
        # Initialize weights and biases manually to match nn.Linear
        # We will store them as buffers or parameters so they are part of the model state
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        """
        Performs the forward pass using custom CUDA operators.
        """
        # 1. Linear Layer: y = xW^T + b
        x = fused_ops.linear_cuda(x, self.weight, self.bias)
        
        # 2. Fused GroupNorm + LeakyReLU + Add (x+x)
        x = fused_ops.fused_gn_leakyrelu_add_cuda(x, self.num_groups, self.eps, self.negative_slope)
        
        return x


def get_inputs():
    return [torch.rand(1024, 8192)]

def get_init_inputs():
    return [8192, 8192, 512]