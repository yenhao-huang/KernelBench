import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation:
# Conv2d -> GroupNorm -> Tanh -> HardSwish -> Residual Add -> LogSumExp
# This fusion avoids multiple memory reads/writes between these stages.

custom_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Helper for atomic max if needed, but here we just need sum exp
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Kernel to perform: 
// 1. Conv2d (implied input is already convolved output x_conv)
// 2. GroupNorm on x_conv
// 3. Tanh(x_norm)
// 4. HardSwish(Tanh(x_norm))
// 5. Add residual: x_res = x_conv + HardSwish(...)
// 6. LogSumExp over dim=1 (channels) for each spatial location and batch item

__global__ void fused_ops_kernel(
    const float* __restrict__ x_conv,      // Input to norm, also residual source
    const float* __restrict__ x_residual,  // Original input x for residual addition
    float* __restrict__ out_logsumexp,     // Output tensor [B, 1, H, W]
    int batch_size,
    int in_channels,       // Actually out_channels of conv, let's call it C
    int height,
    int width,
    int groups,
    float eps
) {
    // Each thread block handles one spatial location (h, w) across all batches and channels?
    // Or better: Each thread handles one element in the output [B, 1, H, W].
    // Total elements = B * H * W.
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * height * width;
    
    if (idx >= total_elements) return;
    
    int b = idx / (height * width);
    int hw_idx = idx % (height * width);
    int h = hw_idx / width;
    int w = hw_idx % width;
    
    // We need to compute LogSumExp over the channel dimension C for this specific (b, h, w)
    // The value at channel c is: x_res[b, c, h, w]
    // where x_res = x_conv + HardSwish(GroupNorm(x_conv))
    
    // However, GroupNorm depends on statistics of the whole group. 
    // To do this efficiently in a single kernel without shared memory complexity for stats,
    // we might need two passes or careful indexing. 
    // Given the constraint of "inline" and simplicity, let's assume we can compute GN stats 
    // per thread block if we organize threads well, OR we accept that GN is expensive to fuse 
    // perfectly without shared memory reduction.
    
    // Alternative Strategy for Fusion:
    // Since GN requires mean/var over groups, and HardSwish/Tanh are element-wise, 
    // and Residual is element-wise, and LogSumExp is reduction.
    // A true single-kernel fusion of GN + Elementwise + Reduction is complex due to the 
    // global reduction nature of GN stats vs local elementwise ops.
    
    // Let's split into two kernels for better optimization:
    // 1. Fused Conv+GN+Tanh+HardSwish+Residual -> Intermediate Tensor
    // 2. LogSumExp kernel
    
    // But the prompt asks to replace operators. We can replace the whole forward logic 
    // with a custom function if we load it. However, PyTorch's nn.Module structure 
    // expects standard calls. The example shows replacing specific ops.
    
    // Let's implement a fused kernel for: GroupNorm -> Tanh -> HardSwish -> Residual Add
    // And keep LogSumExp as a standard torch call or fuse it if easy.
    // Actually, let's try to fuse GN + Elementwise + Residual into one kernel 
    // that outputs the residual tensor, then use torch.logsumexp on it.
    
    // To make this robust and compilable without complex shared memory reductions for GN stats:
    // We will implement a kernel that computes GN statistics in global memory (slow but correct)
    // or assume small groups. 
    // Better: Use the standard approach where we compute mean/var per group.
    
    // Let's define a simpler fused kernel: Conv is already done. 
    // We take x_conv and x_residual (original input).
    // We compute GN(x_conv), then Tanh, HardSwish, Add to x_residual.
    
    // Since GN stats depend on the whole group, we can't easily do it per-thread without 
    // atomic adds or shared memory. 
    // Let's use a two-step approach within the custom module:
    // 1. A kernel that computes GN Mean/Var (reduction)
    // 2. A kernel that applies GN, Tanh, HardSwish, Residual Add
    
    // This is getting too complex for a single inline block without helper functions.
    
    // Let's stick to replacing the most expensive parts: 
    // 1. Conv2d (using cuDNN via torch is usually best, but we can write a simple im2col+gemm if needed. 
    //    However, standard conv is highly optimized. Let's leave Conv alone or use a simple one.)
    // 2. The sequence GN -> Tanh -> HardSwish -> Residual Add.
    
    // Actually, the most significant speedup often comes from fusing element-wise ops 
    // and avoiding global memory writes between them.
    
    // Let's implement a kernel for: 
    // out[b, c, h, w] = x_residual[b, c, h, w] + HardSwish(Tanh(GN(x_conv)[b, c, h, w]))
    
    // To compute GN correctly in a kernel:
    // We need mean and var for each group.
    // Group size G = C / groups.
    // For each group g, we iterate over all channels c in [g*G, (g+1)*G] for all b,h,w.
    
    // This requires a reduction pass. 
    // Let's provide a kernel that does the element-wise part assuming mean/var are precomputed?
    // No, we need to replace the operator.
    
    // Let's use a pragmatic approach:
    // 1. Compute GN stats using a custom reduction kernel (or just call torch functions for stats if allowed? 
    //    The prompt says "replace pytorch operators". We can replace GroupNorm with a custom one.)
    
    // Custom GroupNorm Kernel:
    // Step 1: Reduce to find mean and var for each group.
    // Step 2: Normalize, Tanh, HardSwish, Add Residual.
    
    // Due to complexity of writing a full GN from scratch in inline CUDA with shared memory 
    // optimizations in this format, I will provide a simplified but functional version 
    // that uses global memory reductions for stats (slower than shared mem but correct and simple)
    // OR I will use the fact that we can call torch functions inside the kernel? No.
    
    // Let's write a kernel that performs the element-wise operations: Tanh, HardSwish, Residual Add.
    // And assume GN is handled by a separate custom kernel or standard call? 
    // The prompt allows replacing "some operators".
    
    // I will replace:
    // 1. GroupNorm with a custom CUDA implementation (simplified stats calculation)
    // 2. Tanh + HardSwish + Residual Add into a single fused kernel
    
    // Actually, let's just fuse GN + Tanh + HardSwish + Residual Add into one kernel 
    // using a naive global reduction for stats to ensure correctness and compilation success.
    
    int group_size = in_channels / groups;
    
    // Calculate mean and var for the group this channel belongs to
    int g = (idx % in_channels) / group_size; // This is wrong, idx is linear index over B*H*W
    
    // Let's restructure: Each thread handles one (b, h, w).
    // It needs to process all C channels? No, that's too much work per thread.
    
    // Better: Each thread block handles one (b, h, w) and all C channels.
    // Block size = 256. If C > 256, we need multiple blocks or threads per channel.
    
    // Let's assume C=64, which fits in a block.
    
    extern __shared__ float shared_mem[];
    
    int tid = threadIdx.x;
    int b_local = b;
    int h_local = h;
    int w_local = w;
    
    // Pointer to the data for this (b, h, w) across all channels
    // x_conv[b, :, h, w] is contiguous in memory if channel last? No, channel first.
    // Stride: C * H * W. 
    // Offset for (b, h, w): b*C*H*W + h*W + w
    
    float base_offset = (b_local * height * width + h_local * width + w_local);
    
    // Load all channels for this spatial location into shared memory
    if (tid < in_channels) {
        shared_mem[tid] = x_conv[base_offset + tid * (height * width)];
    }
    __syncthreads();
    
    // Compute mean and var for each group
    // We have 'groups' groups. Each group has 'group_size' channels.
    // We can compute stats in shared memory if we organize threads by group?
    // Or just use global atomic adds? No, too slow.
    
    // Let's use a simple approach: 
    // If C is small (64), we can do it in registers/shared mem easily.
    
    float mean = 0.0f;
    float var_sum = 0.0f;
    
    // Compute mean for the group of channel tid
    int g_idx = tid / group_size;
    float local_mean = 0.0f;
    float local_var_sum = 0.0f;
    
    // Iterate over all channels in this group to compute stats
    // This is inefficient if done per thread, but correct.
    // Better: One thread per group computes stats for that group.
    
    if (tid < groups) {
        float sum = 0.0f;
        float sq_sum = 0.0f;
        int start_c = tid * group_size;
        for (int c = 0; c < group_size; ++c) {
            float val = shared_mem[start_c + c];
            sum += val;
            sq_sum += val * val;
        }
        // Reduce sum within block? No, each thread handles one group.
        // We need to store these stats.
        // Let's use a global array for stats? Or shared memory array of size groups*2.
        
        // Since we are in a kernel with shared memory, let's assume we have enough shared mem.
        // But we can't easily return values from this "thread" to others without sync.
    }
    
    // This is getting too complicated for a single inline block without careful shared memory management.
    // Let's simplify: 
    // 1. Use standard PyTorch GroupNorm (it's already optimized).
    // 2. Fuse Tanh, HardSwish, and Residual Addition into a single kernel.
    // 3. Keep LogSumExp as is or fuse it if easy.
    
    // This is a valid optimization strategy: "replace some operators".
    
    return; 
}

// Kernel for Tanh + HardSwish + Residual Add
__global__ void tanh_hardswish_residual_kernel(
    const float* __restrict__ x_norm,      // Output of GroupNorm
    const float* __restrict__ x_residual,  // Original input (x_conv)
    float* __restrict__ out,               // Result: x_res
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float n = x_norm[idx];
        float r = x_residual[idx];
        
        // Tanh(n)
        float t = tanhf(n);
        
        // HardSwish(t) = t * ReLU6(t+3)/6
        // ReLU6(x) = min(max(x, 0), 6)
        float t_plus_3 = t + 3.0f;
        float relu6 = fminf(fmaxf(t_plus_3, 0.0f), 6.0f);
        float hs = t * (relu6 / 6.0f);
        
        out[idx] = r + hs;
    }
}

// Kernel for LogSumExp
__global__ void logsumexp_kernel(
    const float* __restrict__ x,           // Input [B, C, H, W]
    float* __restrict__ out,               // Output [B, 1, H, W]
    int batch_size,
    int channels,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * height * width;
    
    if (idx >= total_elements) return;
    
    int b = idx / (height * width);
    int hw_idx = idx % (height * width);
    
    // Find max over channels for this (b, h, w)
    float max_val = -INFINITY;
    for (int c = 0; c < channels; ++c) {
        int offset = b * channels * height * width + c * height * width + hw_idx;
        if (x[offset] > max_val) {
            max_val = x[offset];
        }
    }
    
    // Compute sum of exp(x - max)
    float sum_exp = 0.0f;
    for (int c = 0; c < channels; ++c) {
        int offset = b * channels * height * width + c * height * width + hw_idx;
        sum_exp += expf(x[offset] - max_val);
    }
    
    out[idx] = logf(sum_exp) + max_val;
}

torch::Tensor fused_tanh_hardswish_residual_cuda(
    torch::Tensor x_norm, 
    torch::Tensor x_residual
) {
    auto size = x_norm.numel();
    auto out = torch::empty_like(x_norm);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    tanh_hardswish_residual_kernel<<<num_blocks, block_size>>>(
        x_norm.data_ptr<float>(), 
        x_residual.data_ptr<float>(), 
        out.data_ptr<float>(), 
        size
    );
    
    return out;
}

torch::Tensor logsumexp_cuda(torch::Tensor x) {
    // x is [B, C, H, W]
    auto batch_size = x.size(0);
    auto channels = x.size(1);
    auto height = x.size(2);
    auto width = x.size(3);
    
    auto out_shape = torch::IntArrayRef({batch_size, 1, height, width});
    auto out = torch::empty(out_shape, x.options());
    
    int total_elements = batch_size * height * width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    logsumexp_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width
    );
    
    return out;
}
"""

custom_cpp_source = (
    "torch::Tensor fused_tanh_hardswish_residual_cuda(torch::Tensor x_norm, torch::Tensor x_residual);"
    "torch::Tensor logsumexp_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code
custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=custom_cpp_source,
    cuda_sources=custom_cuda_source,
    functions=["fused_tanh_hardswish_residual_cuda", "logsumexp_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"]
)


class ModelNew(nn.Module):
    """
    Optimized Model using custom CUDA operators for:
    1. Fused Tanh + HardSwish + Residual Addition
    2. Custom LogSumExp
    GroupNorm and Conv are left to PyTorch's optimized implementations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        # We don't need Tanh and HardSwish modules anymore as they are fused

    def forward(self, x):
        # Convolution (Standard PyTorch)
        x_conv = self.conv(x)
        
        # Group Normalization (Standard PyTorch)
        x_norm = self.group_norm(x_conv)
        
        # Fused Tanh + HardSwish + Residual Addition
        # Input to fused kernel: normalized output and original conv output for residual
        x_res = custom_ops.fused_tanh_hardswish_residual_cuda(x_norm, x_conv)
        
        # Custom LogSumExp
        x_logsumexp = custom_ops.logsumexp_cuda(x_res)
        
        return x_logsumexp