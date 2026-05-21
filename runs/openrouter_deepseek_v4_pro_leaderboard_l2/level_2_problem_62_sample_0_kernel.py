import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel source
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_groupnorm_leakyrelu_scale_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int N, int C, int num_groups, float eps, float negative_slope)
{
    int group_idx = blockIdx.x;
    int sample_idx = blockIdx.y;
    int group_size = C / num_groups;
    int tid = threadIdx.x;
    
    // Shared memory for reduction (max group_size assumed <= 32)
    __shared__ float s_mean[32];
    __shared__ float s_var[32];
    
    int offset = sample_idx * C + group_idx * group_size;
    float val = x[offset + tid];
    
    // Compute sum for mean
    s_mean[tid] = val;
    __syncthreads();
    
    for (int s = group_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_mean[tid] += s_mean[tid + s];
        }
        __syncthreads();
    }
    float mean = s_mean[0] / group_size;
    
    // Compute variance
    float diff = val - mean;
    s_var[tid] = diff * diff;
    __syncthreads();
    
    for (int s = group_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_var[tid] += s_var[tid + s];
        }
        __syncthreads();
    }
    float var = s_var[0] / group_size;
    
    // Normalize
    float inv_std = rsqrtf(var + eps);
    float norm = diff * inv_std;
    
    // Affine transformation
    int gamma_offset = group_idx * group_size + tid;
    norm = norm * gamma[gamma_offset] + beta[gamma_offset];
    
    // Leaky ReLU
    norm = norm >= 0.0f ? norm : norm * negative_slope;
    
    // Scale by 2 (equivalent to x + x)
    norm *= 2.0f;
    
    // Store result
    out[offset + tid] = norm;
}

torch::Tensor fused_groupnorm_leakyrelu_scale_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    int num_groups,
    float eps,
    float negative_slope)
{
    int N = x.size(0);
    int C = x.size(1);
    int group_size = C / num_groups;
    
    auto out = torch::empty_like(x);
    
    dim3 grid(num_groups, N);
    dim3 block(group_size);
    
    fused_groupnorm_leakyrelu_scale_kernel<<<grid, block>>>(
        x.data_ptr<float>(), out.data_ptr<float>(),
        gamma.data_ptr<float>(), beta.data_ptr<float>(),
        N, C, num_groups, eps, negative_slope);
    
    return out;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_groupnorm_leakyrelu_scale_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    int num_groups,
    float eps,
    float negative_slope);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_groupnorm_leakyrelu_scale",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_groupnorm_leakyrelu_scale_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super(ModelNew, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        # Parameters to replace GroupNorm affine
        self.gn_weight = nn.Parameter(torch.ones(hidden_size))
        self.gn_bias = nn.Parameter(torch.zeros(hidden_size))
        self.num_groups = num_groups
        self.eps = eps
        self.negative_slope = negative_slope
        self.fused_op = fused_op

    def forward(self, x):
        x = self.fc(x)  # (batch_size, hidden_size)
        x = self.fused_op.fused_groupnorm_leakyrelu_scale_cuda(
            x, self.gn_weight, self.gn_bias,
            self.num_groups, self.eps, self.negative_slope
        )
        return x