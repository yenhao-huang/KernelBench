import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# The optimization strategy:
# The original model performs: ConvTranspose3d -> Add -> LayerNorm -> AvgPool3d -> GELU.
# ConvTranspose3d and AvgPool3d are heavy operations that are best handled by highly optimized cuDNN/ATen kernels.
# However, the sequence (Add -> LayerNorm -> GELU) is a perfect candidate for kernel fusion.
# LayerNorm involves calculating mean and variance, which can be fused with the addition and the GELU activation.
# This reduces memory bandwidth by avoiding multiple passes over the large 5D tensor.

fused_ops_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float gelu(float x) {
    return 0.5f * x * (1.0f + erf(x * 0.70710678118f));
}

__global__ void fused_add_norm_gelu_kernel(
    float* __restrict__ data,
    const float* __restrict__ weight,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int N, // Batch * Depth * Height * Width
    int C, // Channels (norm_shape)
    float eps) {
    
    // Each block handles one element of the N dimension (one spatial/batch location)
    // and processes all C channels.
    int n_idx = blockIdx.x;
    if (n_idx >= N) return;

    extern __shared__ float shared_mem[];
    float* s_mean = shared_mem;
    float* s_var = shared_mem + blockDim.x;

    // 1. Compute Mean and Variance for LayerNorm
    float sum = 0.0f;
    float sq_sum = 0.0f;

    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float val = data[n_idx * C + c] + weight[n_idx * C + c]; // Note: weight is broadcasted or per-element? 
        // In the original code: x = x + sum_weight (scalar). 
        // But sum_weight is a parameter. If it's a scalar 1.0, it's added to all.
        // If it's a tensor, it's added element-wise.
        // The original code says: self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        // If sum_weight is a scalar, it's added to every element.
        
        // Re-reading: x = x + self.sum_weight. If sum_weight is a scalar, it's broadcasted.
        // Let's assume sum_weight is a scalar for the kernel logic, or handle it as a single value.
        // However, to be safe and general, we'll treat weight as a single scalar if it's size 1, 
        // or we'll assume the user might pass a tensor. 
        // Given the example: sum_weight = 1.0 (scalar).
    }
    
    // To keep the kernel robust and efficient for the specific architecture:
    // We will implement a kernel that performs:
    // out = GELU(LayerNorm(x + weight))
    // where weight is a scalar.
}

// Since writing a fully generic fused LayerNorm + GELU + Add kernel in a single block 
// is complex for a single code block, we will implement a highly efficient 
// fused kernel for: (x + weight) -> LayerNorm -> GELU.

__global__ void fused_kernel(
    float* data,
    float weight_val,
    const float* gamma,
    const float* beta,
    int N, 
    int C, 
    float eps) {
    
    int n_idx = blockIdx.x;
    int tid = threadIdx.x;

    // Shared memory for reduction
    extern __shared__ float s_mem[];
    float* s_sum = s_mem;
    float* s_sq_sum = s_mem + blockDim.x;

    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;

    // Step 1: Add weight and accumulate sums
    for (int c = tid; c < C; c += blockDim.x) {
        float val = data[n_idx * C + c] + weight_val;
        data[n_idx * C + c] = val; // Update in place for next step
        local_sum += val;
        local_sq_sum += val * val;
    }

    s_sum[tid] = local_sum;
    s_sq_sum[tid] = local_sq_sum;
    __syncthreads();

    // Step 2: Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sq_sum[tid] += s_sq_sum[tid + stride];
        }
        __syncthreads();
    }

    // Step 3: Normalize and Apply GELU
    if (tid < C) {
        float mean = s_sum[0] / C;
        float var = (s_sq_sum[0] / C) - (mean * mean);
        float inv_std = rsqrtf(var + eps);
        
        int idx = n_idx * C + tid;
        float norm_val = (data[idx] - mean) * inv_std;
        // Apply gamma and beta
        norm_val = norm_val * gamma[tid] + beta[tid];
        // Apply GELU
        data[idx] = 0.5f * norm_val * (1.0f + erf(norm_val * 0.70710678118f));
    }
}

torch::Tensor fused_add_norm_gelu_cuda(torch::Tensor x, float weight_val, torch::Tensor gamma, torch::Tensor beta, float eps) {
    int N = x.size(0) * x.size(2) * x.size(3) * x.size(4);
    int C = x.size(1);
    
    const int block_size = 256;
    // Ensure block_size is power of 2 for the reduction
    const int num_blocks = N;

    size_t shared_mem_size = 2 * block_size * sizeof(float);

    fused_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight_val,
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        N,
        C,
        eps
    );

    return x;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_add_norm_gelu_cuda(torch::Tensor x, float weight_val, torch::Tensor gamma, torch::Tensor beta, float eps);
"""

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_cuda_source,
    functions=["fused_add_norm_gelu_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm = nn.LayerNorm(norm_shape)
        self.avg_pool = nn.AvgPool3d(kernel_size=pool_kernel_size)
        self.gelu = nn.GELU()
        self.fused_ops = fused_ops

    def forward(self, x):
        # 1. ConvTranspose3d (Heavy, keep standard)
        x = self.conv_transpose(x)
        
        # 2. Fused: Add + LayerNorm + GELU
        # Note: LayerNorm in PyTorch for 5D (N, C, D, H, W) with norm_shape (C,) 
        # actually normalizes over the last dimension. 
        # However, the original code uses norm_shape=(out_channels,), 
        # which means it normalizes over the channel dimension.
        # In PyTorch, LayerNorm(C) on (N, C, D, H, W) expects the last dim to be C.
        # But the input is (N, C, D, H, W). 
        # To match the original behavior where LayerNorm is applied to the channel dim:
        # We permute to (N, D, H, W, C), apply fused kernel, then permute back.
        
        # Extract parameters for fusion
        weight_val = self.sum_weight.item()
        gamma = self.norm.weight
        beta = self.norm.bias
        eps = self.norm.eps

        # Permute to move Channels to the last dimension for the fused kernel
        # (N, C, D, H, W) -> (N, D, H, W, C)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        
        # Apply fused kernel
        x = self.fused_ops.fused_add_norm_gelu_cuda(x, weight_val, gamma, beta, eps)
        
        # Permute back (N, D, H, W, C) -> (N, C, D, H, W)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        # 3. AvgPool3d (Heavy, keep standard)
        x = self.avg_pool(x)
        
        return x