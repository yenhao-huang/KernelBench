import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source code for fused post-convolution operations
fused_post_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void spatial_mean_bias_kernel(const float* __restrict__ x, const float* __restrict__ bias, float* __restrict__ out,
                                         int N, int C, int H, int W) {
    int idx = blockIdx.x; // idx = n*C + c
    int n = idx / C;
    int c = idx % C;
    int spatial_size = H * W;
    
    // Each thread computes partial sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        int h = i / W;
        int w = i % W;
        sum += x[((n * C + c) * H + h) * W + w];
    }
    
    // Block reduction to get total sum
    __shared__ float sdata[256]; // blockDim.x assumed 256
    sdata[threadIdx.x] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        float mean = sdata[0] / spatial_size;
        out[n * C + c] = mean + bias[c];
    }
}

__global__ void logsumexp_channel_kernel(const float* __restrict__ x, float* __restrict__ out,
                                         int N, int C, float multiplier) {
    int n = blockIdx.x;
    extern __shared__ float shared_data[]; // size 2*C: first C for max, next C for sum
    float* s_max = shared_data;
    float* s_sum = shared_data + C;
    
    int tid = threadIdx.x;
    float val = (tid < C) ? x[n * C + tid] : -INFINITY;
    s_max[tid] = val;
    __syncthreads();
    
    // Find max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }
    float max_val = s_max[0];
    
    // Compute exp and sum
    float exp_val = (tid < C) ? expf(x[n * C + tid] - max_val) : 0.0f;
    s_sum[tid] = exp_val;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        float logsumexp = logf(s_sum[0]) + max_val;
        out[n] = logsumexp * multiplier;
    }
}

torch::Tensor fused_post_conv_cuda(torch::Tensor x, torch::Tensor bias, float multiplier) {
    // x: (N, C, H, W) output of conv transpose
    // bias: (C,) or (C,1,1) we'll flatten
    auto N = x.size(0);
    auto C = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);
    
    // Ensure bias is 1D
    auto bias_flat = bias.view({-1});
    TORCH_CHECK(bias_flat.size(0) == C, "Bias size mismatch");
    
    // Allocate intermediate buffer for per-channel means (N, C)
    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto temp = torch::empty({N, C}, options);
    
    // Launch spatial mean + bias kernel
    const int block_size = 256;
    const int grid_size = N * C;
    spatial_mean_bias_kernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(), bias_flat.data_ptr<float>(), temp.data_ptr<float>(),
        N, C, H, W);
    
    // Allocate output (N, 1)
    auto out = torch::empty({N, 1}, options);
    
    // Launch logsumexp kernel
    const int block_size2 = C; // 128
    const int grid_size2 = N;
    // Shared memory size: 2 * C * sizeof(float)
    size_t shared_mem_size = 2 * C * sizeof(float);
    logsumexp_channel_kernel<<<grid_size2, block_size2, shared_mem_size>>>(
        temp.data_ptr<float>(), out.data_ptr<float>(),
        N, C, multiplier);
    
    return out;
}
"""

fused_post_conv_cpp_source = "torch::Tensor fused_post_conv_cuda(torch::Tensor x, torch::Tensor bias, float multiplier);"

# Compile the inline CUDA code
fused_post_conv = load_inline(
    name="fused_post_conv",
    cpp_sources=fused_post_conv_cpp_source,
    cuda_sources=fused_post_conv_source,
    functions=["fused_post_conv_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, then uses a fused CUDA kernel
    for global average pooling, bias addition, log-sum-exp, sum, and multiplication.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_post_conv = fused_post_conv

    def forward(self, x):
        x = self.conv_transpose(x)
        # Fused kernel: global average pooling + bias + logsumexp + sum + multiply
        x = self.fused_post_conv.fused_post_conv_cuda(x, self.bias, 10.0)
        return x