import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused: Bias-addition + GroupNorm + LeakyReLU + Elementwise Sum
# Note: MatMul is handled by cuBLAS for maximum performance.
# This kernel handles the pointwise and reduction-based operations following the MatMul.
# For GroupNorm on (N, C) where C is the hidden dimension, each group has C/num_groups elements.
# The kernel calculates mean and variance per group per batch element, then applies normalization,
# LeakyReLU, and the final element-wise sum (x + x).

fused_ops_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_gn_leaky_sum_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int num_channels,
    int num_groups,
    float eps,
    float negative_slope) {

    int b = blockIdx.x; // Each block handles one batch element
    int group_idx = blockIdx.y;
    int tid = threadIdx.x;

    int channels_per_group = num_channels / num_groups;
    int group_start_idx = group_idx * channels_per_group;
    
    // 1. Calculate Mean and Variance for the group
    // We use a simple reduction for clarity; in production, one might use warp shuffles.
    extern __shared__ float shared_mem[];
    float* s_sum = shared_mem;
    float* s_sq_sum = &shared_mem[blockDim.x];

    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;

    for (int i = tid; i < channels_per_group; i += blockDim.x) {
        float val = input[b * num_channels + group_start_idx + i];
        local_sum += val;
        local_sq_sum += val * val;
    }

    // Block-level reduction
    s_sum[tid] = local_sum;
    s_sq_sum[tid] = local_sq_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sq_sum[tid] += s_sq_sum[tid + s];
        }
        __syncthreads();
    }

    float group_mean = s_sum[0] / channels_per_group;
    float group_var = (s_sq_sum[0] / channels_per_group) - (group_mean * group_mean);
    float inv_std = 1.0f / sqrtf(group_var + eps);

    // 2. Apply GroupNorm, LeakyReLU, and Sum (x + x)
    // We use the weight and bias from GroupNorm (which are 1D tensors of size num_channels)
    // and the bias from the Linear layer.
    // Note: The input to this kernel is the result of (MatMul + Linear_Bias).
    
    for (int i = tid; i < channels_per_group; i += blockDim.x) {
        int idx = b * num_channels + group_start_idx + i;
        float val = input[idx];
        
        // GroupNorm: (val - mean) * inv_std * weight + bias
        // Note: GroupNorm in PyTorch uses weight/bias of shape (C,)
        float norm_val = (val - group_mean) * inv_std * weight[group_start_idx + i] + bias[group_start_idx + i];
        
        // LeakyReLU
        float leaky_val = (norm_val > 0) ? norm_val : norm_val * negative_slope;
        
        // Element-wise sum: x + x
        output[idx] = leaky_val + leaky_val;
    }
}

torch::Tensor fused_gn_leaky_sum_cuda(
    torch::Tensor input, 
    torch::Tensor gn_weight, 
    torch::Tensor gn_bias, 
    int num_groups, 
    float eps, 
    float negative_slope) {
    
    int batch_size = input.size(0);
    int num_channels = input.size(1);
    auto output = torch::empty_like(input);

    dim3 grid(batch_size, num_groups);
    dim3 block(256);
    size_t shared_mem_size = 2 * 256 * sizeof(float);

    fused_gn_leaky_sum_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        gn_weight.data_ptr<float>(),
        gn_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_channels,
        num_groups,
        eps,
        negative_slope
    );

    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_gn_leaky_sum_cuda(
    torch::Tensor input, 
    torch::Tensor gn_weight, 
    torch::Tensor gn_bias, 
    int num_groups, 
    float eps, 
    float negative_slope);
"""

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_cuda_source,
    functions=["fused_gn_leaky_sum_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.eps = eps
        self.negative_slope = negative_slope
        
        self.fc = nn.Linear(input_size, hidden_size)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size, eps=eps)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.fused_ops = fused_ops

    def forward(self, x):
        # 1. Perform Linear (MatMul + Bias)
        # We keep MatMul in PyTorch to leverage highly optimized cuBLAS
        x = self.fc(x)
        
        # 2. Fuse GroupNorm + LeakyReLU + (x + x)
        # We pass the result of the linear layer into our custom kernel
        x = self.fused_ops.fused_gn_leaky_sum_cuda(
            x, 
            self.gn.weight, 
            self.gn.bias, 
            self.num_groups, 
            self.eps, 
            self.negative_slope
        )
        return x