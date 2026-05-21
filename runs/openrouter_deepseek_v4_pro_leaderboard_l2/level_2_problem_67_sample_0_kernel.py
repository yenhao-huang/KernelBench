import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GELU + global average pooling
fused_gelu_avg_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_gelu_avg_pool_kernel(const float* input, float* output, int N, int C, int H, int W) {
    int n = blockIdx.x;
    int c = blockIdx.y;
    int spatial_size = H * W;
    int tid = threadIdx.x;
    
    extern __shared__ float sdata[];
    
    float sum = 0.0f;
    // Constants for GELU approximation
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    
    // Each thread processes multiple elements
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int h = i / W;
        int w = i % W;
        int idx = ((n * C + c) * H + h) * W + w;
        float x = input[idx];
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float x3 = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x3);
        float tanh_inner = tanhf(inner);
        float gelu_x = 0.5f * x * (1.0f + tanh_inner);
        sum += gelu_x;
    }
    
    // Block reduction
    sdata[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[n * C + c] = sdata[0] / spatial_size;
    }
}

torch::Tensor fused_gelu_avg_pool_cuda(torch::Tensor input) {
    // input shape: [N, C, H, W]
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    
    auto output = torch::empty({N, C}, input.options());
    
    const int block_size = 256;
    dim3 grid(N, C);
    
    size_t shared_mem_size = block_size * sizeof(float);
    
    fused_gelu_avg_pool_kernel<<<grid, block_size, shared_mem_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), N, C, H, W);
    
    return output;
}
"""

fused_gelu_avg_pool_cpp_source = "torch::Tensor fused_gelu_avg_pool_cuda(torch::Tensor input);"

# Compile the inline CUDA code
fused_gelu_avg_pool = load_inline(
    name="fused_gelu_avg_pool",
    cpp_sources=fused_gelu_avg_pool_cpp_source,
    cuda_sources=fused_gelu_avg_pool_source,
    functions=["fused_gelu_avg_pool_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model: convolution followed by fused GELU + global average pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.fused_gelu_avg_pool = fused_gelu_avg_pool

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
        Returns:
            Output tensor of shape (batch_size, out_channels)
        """
        x = self.conv(x)
        x = self.fused_gelu_avg_pool.fused_gelu_avg_pool_cuda(x)
        return x


# The following inputs are kept for compatibility with the original interface
batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]