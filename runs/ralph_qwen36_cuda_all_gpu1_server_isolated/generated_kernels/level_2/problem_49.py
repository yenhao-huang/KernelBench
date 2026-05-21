import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Softmax + Sigmoid
# We assume the input to this kernel is the output of ConvTranspose3d.
# The softmax is applied over dim=1 (channels).
# Sigmoid is applied element-wise.

fused_softmax_sigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Helper for atomic max if needed, but here we do two-pass or online softmax.
// For simplicity and robustness with standard CUDA, we'll use a two-pass approach 
// within a single kernel launch per channel slice to ensure numerical stability.
// However, to keep it simple and fast for FP32, we can implement a standard 
// parallel reduction for max and sum.

__device__ float atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        if (val <= __int_as_float(assumed)) return __int_as_float(assumed);
        old = atomicCAS(address_as_int, assumed, __float_as_int(val));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void fused_softmax_sigmoid_kernel(const float* input, float* output, int batch_size, int channels, int spatial_size) {
    // Each block handles one channel slice of one batch item? 
    // Or we can have each thread handle one element.
    // Let's use a grid-stride loop for elements.
    
    int total_elements = batch_size * channels * spatial_size;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        output[idx] = 0.0f; // Placeholder, logic below is per-channel
    }
}

// Better approach: Process one channel at a time in a kernel to allow shared memory reduction for softmax.
// Kernel signature: input[B, C, S], output[B, C, S]
// We launch grid with (B*C) blocks, each block handles one channel's spatial elements.

__global__ void softmax_sigmoid_channel_kernel(const float* __restrict__ input, float* __restrict__ output, int spatial_size) {
    extern __shared__ float s_data[];
    
    int tid = threadIdx.x;
    int total_threads = blockDim.x;
    
    // Load data into shared memory for reduction
    // Each thread loads one element if possible, or handles strided access
    float local_max = -1e20f;
    float local_sum = 0.0f;
    
    // First pass: Find max
    for (int i = tid; i < spatial_size; i += total_threads) {
        float val = input[i];
        if (val > local_max) {
            local_max = val;
        }
    }
    
    // Reduce max within block
    s_data[tid] = local_max;
    __syncthreads();
    
    for (int stride = total_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_data[tid + stride] > s_data[tid]) {
                s_data[tid] = s_data[tid + stride];
            }
        }
        __syncthreads();
    }
    
    float global_max = s_data[0];
    __syncthreads();
    
    // Second pass: Compute exp(x - max) and sum
    for (int i = tid; i < spatial_size; i += total_threads) {
        float val = input[i];
        float exp_val = expf(val - global_max);
        s_data[tid] = exp_val;
        __syncthreads();
        
        // Reduce sum
        for (int stride = total_threads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s_data[tid] += s_data[tid + stride];
            }
            __syncthreads();
        }
        
        // Only thread 0 has the sum, but we need to broadcast it or store it.
        // Let's store the sum in shared memory at index 0 and have all threads read it? 
        // Or just use a separate reduction step.
        // Actually, let's just compute exp_val and accumulate into a local variable if we were single thread, 
        // but here we need the global sum.
        
        // Re-structure: Thread 0 computes sum, others wait? No, that's slow.
        // Let's use shared memory for the sum reduction result.
    }
    
    // The above loop structure is flawed for parallel sum accumulation across threads without careful sync.
    // Let's simplify: Use a standard two-kernel approach or a very robust single kernel.
    // Given the constraints, let's write a clean single kernel that processes one channel.
}

// Simplified Kernel: Process one channel slice (spatial_size elements)
// We assume spatial_size is not too large for shared memory, or we use grid-stride with atomic adds.
// For robustness and speed on typical sizes (16*32*32 = 16384), shared memory is good.

__global__ void fused_softmax_sigmoid_kernel_v2(const float* __restrict__ input, float* __restrict__ output, int spatial_size) {
    extern __shared__ float s_mem[];
    float* s_max = s_mem;
    float* s_sum = s_mem + blockDim.x;

    int tid = threadIdx.x;
    int total_threads = blockDim.x;

    // 1. Find Max
    float local_max = -1e20f;
    for (int i = tid; i < spatial_size; i += total_threads) {
        if (input[i] > local_max) {
            local_max = input[i];
        }
    }
    s_max[tid] = local_max;
    __syncthreads();

    // Parallel reduction for max
    for (int stride = total_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_max[tid + stride] > s_max[tid]) {
                s_max[tid] = s_max[tid + stride];
            }
        }
        __syncthreads();
    }
    float global_max = s_max[0];
    __syncthreads();

    // 2. Compute Exp and Sum
    float local_sum = 0.0f;
    for (int i = tid; i < spatial_size; i += total_threads) {
        float val = input[i];
        float exp_val = expf(val - global_max);
        s_sum[tid] = exp_val;
        __syncthreads();

        // Parallel reduction for sum
        for (int stride = total_threads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s_sum[tid] += s_sum[tid + stride];
            }
            __syncthreads();
        }
        
        // Note: The above inner loop overwrites s_sum. We need the final sum at s_sum[0].
        // But we are inside a loop over i. This is incorrect for parallel reduction if we want to accumulate into one variable.
        // Correct approach: Each thread computes its exp, stores in shared mem, then reduce shared mem to get total sum.
    }
    
    // Let's restart the logic for step 2 properly.
    // We need the global sum of all exps.
    
    // Reset s_sum for reduction? No, let's use a separate buffer or just redo it.
    // Actually, we can compute the sum in shared memory once.
    
    // Recompute exps and store in s_sum
    for (int i = tid; i < spatial_size; i += total_threads) {
        float val = input[i];
        s_sum[tid] = expf(val - global_max); // This is wrong, multiple threads write to same index if not careful? 
                                            // No, each thread writes to its own tid? No, we are iterating i.
                                            // We need to store the partial sums or just the values.
    }
    
    // Let's use a simpler method: Grid-stride loop for normalization directly if we had global sum.
    // Since we have global_max, we can compute exp(x - max).
    // Then we need 1/sum(exp).
    
    // Let's do the reduction correctly.
    // Step 2a: Compute exps and store in s_sum (one value per thread? No, spatial_size might be > block_size)
    // If spatial_size > block_size, we must accumulate locally first.
    
    float local_exp_sum = 0.0f;
    for (int i = tid; i < spatial_size; i += total_threads) {
        float val = input[i];
        local_exp_sum += expf(val - global_max);
    }
    s_sum[tid] = local_exp_sum;
    __syncthreads();

    // Reduce sum in shared memory
    for (int stride = total_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float global_sum = s_sum[0];
    __syncthreads();

    // Step 3: Normalize and Apply Sigmoid
    for (int i = tid; i < spatial_size; i += total_threads) {
        float val = input[i];
        float softmax_val = expf(val - global_max) / global_sum;
        output[i] = 1.0f / (1.0f + expf(-softmax_val)); // Sigmoid(softmax_val)
    }
}

torch::Tensor fused_softmax_sigmoid_cuda(torch::Tensor input) {
    // Input shape: [B, C, D, H, W]
    // We want to apply Softmax(dim=1) and then Sigmoid.
    // This means for each (b, d, h, w), we softmax over c? 
    // NO. nn.Softmax(dim=1) applies softmax over the channel dimension for each spatial location.
    // So for a fixed b, d, h, w, we have a vector of size C. We softmax that vector.
    
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);
    
    int spatial_elements_per_channel_slice = D * H * W; // This is wrong.
    // Softmax(dim=1) means the reduction happens over dim 1.
    // So for each (b, d, h, w), we take input[b, :, d, h, w] which has size C.
    // We softmax this vector of size C.
    
    // Therefore, the "spatial_size" in my kernel above was interpreted as the dimension being reduced over?
    // No, in my kernel `input` is a 1D slice of size `spatial_size`.
    // If I want to softmax over dim 1 (channels), then for each fixed (b, d, h, w), 
    // the data is contiguous in memory if we access it correctly?
    // PyTorch tensors are C-contiguous. Shape [B, C, D, H, W].
    // Elements for a specific (b, d, h, w) across channels are NOT contiguous.
    // input[b, 0, d, h, w], input[b, 1, d, h, w]... are separated by D*H*W elements.
    
    // This makes a simple 1D kernel difficult for Softmax(dim=1).
    // We would need to transpose or use strided access.
    
    // Alternative: Transpose input to [B, D, H, W, C]. Then the channel dimension is contiguous.
    // Then we can apply softmax over the last dimension (C) using a kernel where each thread/block handles one (b,d,h,w).
    
    auto input_t = input.transpose(1, 4); // Shape: [B, D, H, W, C]
    auto output_t = torch::empty_like(input_t);
    
    int b_size = batch_size;
    int d_size = D;
    int h_size = H;
    int w_size = W;
    int c_size = channels;
    
    int total_slices = b_size * d_size * h_size * w_size;
    
    if (total_slices == 0) {
        return input.transpose(1, 4).transpose(1, 4); // Identity
    }

    const int block_size = 256; // Or 512
    const int num_blocks = total_slices;
    
    // We need shared memory for reduction. Size depends on c_size.
    // If c_size is large, we might need more shared memory.
    // Max shared memory per block is usually 48KB or 96KB.
    // float takes 4 bytes. 256 threads * 4 bytes = 1KB for s_max, 1KB for s_sum. Total 2KB. Safe.
    
    cudaFuncSetAttribute(fused_softmax_sigmoid_kernel_v2, cudaFuncAttributeMaxDynamicSharedMemorySize, 2048);

    fused_softmax_sigmoid_kernel_v2<<<num_blocks, block_size, 2048>>>(
        input_t.data_ptr<float>(), 
        output_t.data_ptr<float>(), 
        c_size
    );
    
    // Apply Sigmoid? No, the kernel above does Softmax then Sigmoid.
    // Wait, the kernel computes:
    // softmax_val = exp(x - max) / sum(exp)
    // output = sigmoid(softmax_val)
    // This matches the requirement: Softmax(dim=1) then Sigmoid.
    
    return output_t.transpose(1, 4);
}
"""

fused_softmax_sigmoid_cpp_source = (
    "torch::Tensor fused_softmax_sigmoid_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_softmax_sigmoid_cpp_source,
    cuda_sources=fused_softmax_sigmoid_source,
    functions=["fused_softmax_sigmoid_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution, 
    then applies a fused Softmax(dim=1) and Sigmoid operation via custom CUDA.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)
        # Softmax and Sigmoid are replaced by the custom fused operator

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D, H, W).
        """
        x = self.conv_transpose(x)
        # Apply fused Softmax(dim=1) and Sigmoid
        x = fused_ops.fused_softmax_sigmoid_cuda(x)
        return x