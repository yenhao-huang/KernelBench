import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Fusing GELU and the element-wise part of GroupNorm.
# GroupNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta
# We will perform: x = GELU(x), then y = gamma * (x - mean) * inv_std + beta

fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float fast_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_gelu_gn_kernel(
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ inv_std,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    int N, int C, int H, int W, int G, int channels_per_group) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * H * W;

    if (idx < total_elements) {
        int w = idx % W;
        int h = (idx / W) % H;
        int c = (idx / (W * H)) % C;
        int n = idx / (C * H * W);

        int group_idx = c / channels_per_group;
        int g_idx = n * G + group_idx;

        float val = x[idx];
        // Apply GELU
        val = fast_gelu(val);
        
        // Apply GroupNorm element-wise
        out[idx] = gamma[c] * (val - mean[g_idx]) * inv_std[g_idx] + beta[c];
    }
}

torch::Tensor fused_gelu_gn_cuda(
    torch::Tensor x, 
    torch::Tensor gamma, 
    torch::Tensor beta, 
    torch::Tensor mean, 
    torch::Tensor inv_std,
    int G, int channels_per_group) {
    
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int total_elements = x.numel();
    
    auto out = torch::empty_like(x);
    
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_gelu_gn_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        mean.data_ptr<float>(),
        inv_std.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W, G, channels_per_group
    );
    
    return out;
}
"""

fused_kernel_cpp_source = "torch::Tensor fused_gelu_gn_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, torch::Tensor mean, torch::Tensor inv_std, int G, int channels_per_group);"

fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_gelu_gn_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.num_groups = num_groups
        self.out_channels = out_channels
        
        # GroupNorm parameters
        self.gamma = nn.Parameter(torch.ones(out_channels))
        self.beta = nn.Parameter(torch.zeros(out_channels))
        self.fused_op = fused_op

    def forward(self, x):
        # 1. ConvTranspose2d (cuDNN optimized)
        x = self.conv_transpose(x)
        
        # 2. Calculate stats for GroupNorm on the GELU-transformed input
        # To match the original Model: x = GN(GELU(x))
        # We first compute GELU to get the values that GN will normalize.
        x_gelu = torch.nn.functional.gelu(x)
        
        N, C, H, W = x_gelu.shape
        G = self.num_groups
        channels_per_group = C // G
        
        # Reshape to [N, G, -1] to compute mean/var per group
        x_reshaped = x_gelu.view(N, G, -1)
        mean = x_reshaped.mean(dim=-1)
        var = x_reshaped.var(dim=-1, unbiased=False)
        inv_std = torch.rsqrt(var + 1e-5)
        
        # 3. Fused Kernel: Applies GELU (redundant here but kept for kernel logic) 
        # and the element-wise part of GroupNorm.
        # Since we already applied GELU to get stats, we pass x_gelu to the kernel.
        # To avoid double GELU, we can modify the kernel or just accept it.
        # Let's pass x (pre-gelu) to the kernel to be efficient.
        
        return self.fused_op.fused_gelu_gn_cuda(
            x, 
            self.gamma, 
            self.beta, 
            mean, 
            inv_std, 
            G, 
            channels_per_group
        )