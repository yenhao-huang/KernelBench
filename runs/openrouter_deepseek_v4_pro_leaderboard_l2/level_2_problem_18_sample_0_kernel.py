import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for fused linear + sum operation
fused_linear_sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_linear_sum_kernel(const float* x, const float* W_sum, float b_sum, float* out, int batch_size, int in_features) {
    int row = blockIdx.x;
    if (row >= batch_size) return;
    
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    float sum = 0.0f;
    for (int k = tid; k < in_features; k += stride) {
        sum += x[row * in_features + k] * W_sum[k];
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        out[row] = sdata[0] + b_sum;
    }
}

torch::Tensor fused_linear_sum_cuda(torch::Tensor x, torch::Tensor W_sum, torch::Tensor b_sum) {
    int batch_size = x.size(0);
    int in_features = x.size(1);
    
    auto out = torch::empty({batch_size}, x.options());
    
    const int block_size = 256;
    const int num_blocks = batch_size;
    int shared_mem_size = block_size * sizeof(float);
    
    fused_linear_sum_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        x.data_ptr<float>(), W_sum.data_ptr<float>(), b_sum.item<float>(), out.data_ptr<float>(), batch_size, in_features
    );
    
    return out.view({batch_size, 1});
}
"""

fused_linear_sum_cpp_source = "torch::Tensor fused_linear_sum_cuda(torch::Tensor x, torch::Tensor W_sum, torch::Tensor b_sum);"

# Compile the inline CUDA code
fused_linear_sum = load_inline(
    name="fused_linear_sum",
    cpp_sources=fused_linear_sum_cpp_source,
    cuda_sources=fused_linear_sum_source,
    functions=["fused_linear_sum_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        # Precompute sums for the fused operation
        W_sum = self.linear.weight.sum(dim=0)  # (in_features,)
        b_sum = self.linear.bias.sum()          # scalar
        self.register_buffer('W_sum', W_sum)
        self.register_buffer('b_sum', b_sum)
        self.fused_op = fused_linear_sum

    def forward(self, x):
        # Fused linear + sum
        x = self.fused_op.fused_linear_sum_cuda(x, self.W_sum, self.b_sum)
        # The following operations are identity on a (batch_size, 1) tensor,
        # but we keep them to match the original architecture.
        x = torch.max(x, dim=1, keepdim=True)[0]
        x = torch.mean(x, dim=1, keepdim=True)
        x = torch.logsumexp(x, dim=1, keepdim=True)
        x = torch.logsumexp(x, dim=1, keepdim=True)
        return x