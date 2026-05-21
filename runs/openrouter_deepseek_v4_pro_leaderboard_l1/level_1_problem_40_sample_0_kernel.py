import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused Layer Normalization
layernorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void layernorm_kernel(const float* __restrict__ x,
                                 const float* __restrict__ weight,
                                 const float* __restrict__ bias,
                                 float* __restrict__ out,
                                 int M, float eps) {
    int sample_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Step 1: compute sum and sum of squares for current sample
    float sum = 0.0f;
    float sq_sum = 0.0f;
    int base_offset = sample_idx * M;
    
    for (int i = tid; i < M; i += block_size) {
        float val = x[base_offset + i];
        sum += val;
        sq_sum += val * val;
    }
    
    // Step 2: block-level reduction using shared memory
    extern __shared__ float shared[];
    float* s_sum = shared;
    float* s_sq = shared + block_size;
    s_sum[tid] = sum;
    s_sq[tid] = sq_sum;
    __syncthreads();
    
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sq[tid] += s_sq[tid + stride];
        }
        __syncthreads();
    }
    
    float mean = s_sum[0] / M;
    float var = s_sq[0] / M - mean * mean;
    float inv_std = rsqrtf(var + eps);
    
    // Step 3: normalize and apply affine transform
    for (int i = tid; i < M; i += block_size) {
        int idx = base_offset + i;
        float normalized = (x[idx] - mean) * inv_std;
        out[idx] = normalized * weight[i] + bias[i];
    }
}

torch::Tensor layernorm_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, float eps) {
    // Ensure contiguous tensors for direct memory access
    x = x.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();
    
    int N = x.size(0);         // batch size
    int M = 1;                 // number of features to normalize over
    for (int i = 1; i < x.dim(); ++i) {
        M *= x.size(i);
    }
    
    auto out = torch::empty_like(x);
    
    const int block_size = 256;
    int shared_mem_bytes = 2 * block_size * sizeof(float);
    
    dim3 grid(N);
    dim3 block(block_size);
    
    layernorm_kernel<<<grid, block, shared_mem_bytes>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        out.data_ptr<float>(), M, eps
    );
    
    return out;
}
"""

layernorm_cpp_source = """
torch::Tensor layernorm_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, float eps);
"""

# Compile the inline CUDA code
layernorm_ops = load_inline(
    name="layernorm_ops",
    cpp_sources=layernorm_cpp_source,
    cuda_sources=layernorm_source,
    functions=["layernorm_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized Layer Normalization model using a custom fused CUDA kernel.
    """
    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        self.eps = 1e-5
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.layernorm_cuda = layernorm_ops.layernorm_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layernorm_cuda(x, self.weight, self.bias, self.eps)