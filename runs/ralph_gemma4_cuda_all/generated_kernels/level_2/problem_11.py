import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel to fuse BatchNorm2d + Tanh
# This kernel performs: out = tanh(x * weight + bias)
# Note: BatchNorm2d in PyTorch uses (x - mean) / sqrt(var + eps) * weight + bias
# We can simplify this to: x * (weight / sqrt(var + eps)) + (bias - mean * weight / sqrt(var + eps))
# Let's call the combined scale 'S' and combined offset 'O'.
# out = tanh(x * S + O)

fused_bn_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_bn_tanh_kernel(
    const float* __restrict__ input,
    const float* __restrict__ scale,
    const float* __restrict__ offset,
    float* __restrict__ output,
    int num_elements) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        // We assume scale and offset are broadcastable over [N, C, H, W]
        // However, for BatchNorm2d, scale and offset are per-channel [C]
        // To make this kernel efficient, we pass the pre-computed per-element scale/offset 
        // or handle the indexing inside. 
        // For simplicity and speed in this specific architecture, we'll assume 
        // the caller provides scale and offset tensors of shape [N, C, 1, 1] or similar.
        // But a more robust way is to pass the channel dimension.
        // Let's assume the scale/offset are already expanded to [N, C, H, W] or we handle it.
        // Actually, let's pass the channel-wise scale/offset and the shape.
        // For this implementation, we'll assume scale/offset are [C] and we handle indexing.
    }
}

// A more practical approach for the inline kernel:
// We will pass the scale and offset as [C] and the input as [N, C, H, W]
__global__ void fused_bn_tanh_kernel_v2(
    const float* __restrict__ input,
    const float* __restrict__ scale,
    const float* __restrict__ offset,
    float* __restrict__ output,
    int N, int C, int H, int W) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    
    if (idx < total) {
        // Calculate channel index
        // idx = n*C*H*W + c*H*W + h*W + w
        int rem = idx % (H * W);
        int c = (idx / (H * W)) % C;
        
        float val = input[idx];
        float s = scale[c];
        float o = offset[c];
        
        output[idx] = tanhf(val * s + o);
    }
}

torch::Tensor fused_bn_tanh_cuda(torch::Tensor input, torch::Tensor scale, torch::Tensor offset) {
    auto N = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);
    auto output = torch::empty_like(input);
    
    int total = N * C * H * W;
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;
    
    fused_bn_tanh_kernel_v2<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        scale.data_ptr<float>(),
        offset.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W);
        
    return output;
}
"""

fused_bn_tanh_cpp_source = "torch::Tensor fused_bn_tanh_cuda(torch::Tensor input, torch::Tensor scale, torch::Tensor offset);"

fused_bn_tanh = load_inline(
    name="fused_bn_tanh",
    cpp_sources=fused_bn_tanh_cpp_source,
    cuda_sources=fused_bn_tanh_source,
    functions=["fused_bn_tanh_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.tanh = nn.Tanh()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.fused_op = fused_bn_tanh

    def forward(self, x):
        # 1. ConvTranspose2d (Standard, highly optimized in cuDNN)
        x = self.conv_transpose(x)
        
        # 2. Fuse BatchNorm2d + Tanh
        # BatchNorm2d: y = (x - running_mean) / sqrt(running_var + eps) * weight + bias
        # Let S = weight / sqrt(running_var + eps)
        # Let O = bias - (running_mean * S)
        # y = x * S + O
        
        # We need to extract parameters from BatchNorm2d
        # Note: In eval mode, we use running_mean/var. In train mode, we use batch stats.
        # For simplicity and to match the user's likely inference-heavy use case, 
        # we implement the fusion for the eval case (using running stats).
        # If training, we'd need a different kernel or just use standard ops.
        
        if self.training:
            x = self.batch_norm(x)
            x = self.tanh(x)
        else:
            # Pre-calculate fused scale and offset
            eps = self.batch_norm.eps
            running_mean = self.batch_norm.running_mean
            running_var = self.batch_norm.running_var
            weight = self.batch_norm.weight
            bias = self.batch_norm.bias
            
            scale = weight / torch.sqrt(running_var + eps)
            offset = bias - (running_mean * scale)
            
            x = self.fused_op.fused_bn_tanh_cuda(x, scale, offset)
            
        # 3. MaxPool2d (Standard)
        x = self.max_pool(x)
        
        # 4. GroupNorm (Standard)
        x = self.group_norm(x)
        
        return x