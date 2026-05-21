import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel to fuse: bias addition and summation along the channel dimension.
# This is applied after the pooling operations.
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_bias_sum_kernel(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int batch_size,
    int channels) {
    
    int b = blockIdx.x;
    if (b >= batch_size) return;

    float sum = 0.0f;
    for (int c = 0; c < channels; ++c) {
        // x is (B, C, 1, 1, 1), bias is (C, 1, 1, 1)
        sum += x[b * channels + c] + bias[c];
    }
    out[b] = sum;
}

torch::Tensor fused_bias_sum_cuda(torch::Tensor x, torch::Tensor bias) {
    int batch_size = x.size(0);
    int channels = x.size(1);
    
    auto out = torch::empty({batch_size}, x.options());

    const int block_size = 1;
    const int num_blocks = batch_size;

    fused_bias_sum_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        channels
    );

    return out;
}
"""

fused_ops_cpp_source = "torch::Tensor fused_bias_sum_cuda(torch::Tensor x, torch::Tensor bias);"

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_bias_sum_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized Model using a custom CUDA kernel to fuse bias addition and summation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim
        self.fused_ops = fused_ops

    def forward(self, x):
        # 1. Convolution (highly optimized via cuDNN)
        x = self.conv(x)
        
        # 2. Element-wise division
        x = x / self.divisor
        
        # 3. Max Pooling (highly optimized via cuDNN)
        x = self.max_pool(x)
        
        # 4. Global Average Pooling
        x = self.global_avg_pool(x)
        
        # x is now (Batch, Channels, 1, 1, 1)
        # 5. Fused Bias Addition and Summation along sum_dim (dim=1)
        # This replaces: x = x + self.bias; x = torch.sum(x, dim=self.sum_dim)
        if self.sum_dim == 1:
            # Flatten x to (B, C) for the kernel
            x_flat = x.view(x.size(0), x.size(1))
            # Flatten bias to (C) for the kernel
            bias_flat = self.bias.view(-1)
            x = self.fused_ops.fused_bias_sum_cuda(x_flat, bias_flat)
        else:
            # Fallback for other sum_dims if necessary, though the prompt specifies sum_dim=1
            x = x + self.bias
            x = torch.sum(x, dim=self.sum_dim)
            
        return x