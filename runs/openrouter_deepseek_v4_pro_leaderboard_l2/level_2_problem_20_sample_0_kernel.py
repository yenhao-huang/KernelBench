import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused operations: bias add, residual add, multiply, and final residual add
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_ops_kernel(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    const float* __restrict__ original_x,
    float* __restrict__ out,
    int batch_size,
    int out_channels,
    int depth,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * depth * height * width;
    
    if (idx < total_elements) {
        // Compute indices
        int w = idx % width;
        int h = (idx / width) % height;
        int d = (idx / (width * height)) % depth;
        int c = (idx / (width * height * depth)) % out_channels;
        int n = idx / (width * height * depth * out_channels);
        
        // Bias index: bias is (out_channels, 1, 1, 1)
        int bias_idx = c;
        
        float x_val = x[idx];
        float orig_val = original_x[idx];
        float bias_val = bias[bias_idx];
        
        // x = x + bias
        float temp1 = x_val + bias_val;
        // x = x + original_x
        float temp2 = temp1 + orig_val;
        // x = x * original_x
        float temp3 = temp2 * orig_val;
        // x = x + original_x
        out[idx] = temp3 + orig_val;
    }
}

torch::Tensor fused_ops_cuda(
    torch::Tensor x,
    torch::Tensor bias,
    torch::Tensor original_x
) {
    auto batch_size = x.size(0);
    auto out_channels = x.size(1);
    auto depth = x.size(2);
    auto height = x.size(3);
    auto width = x.size(4);
    
    auto out = torch::empty_like(x);
    
    int total_elements = batch_size * out_channels * depth * height * width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_ops_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        original_x.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        out_channels,
        depth,
        height,
        width
    );
    
    return out;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor x, torch::Tensor bias, torch::Tensor original_x);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        original_x = x.clone().detach()
        x = self.fused_ops.fused_ops_cuda(x, self.bias, original_x)
        return x