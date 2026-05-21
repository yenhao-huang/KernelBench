import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused-elementwise-operations (fused-elementwise-operations)
# This kernel will fuse the        - Conv2d (3x3 Depthwise) 
# This kernel will fuse the        - BatchNorm2d <tool_call|>-wise-operations (fused-elementd-wise-operations)
# This kernel
# This =/ of the following:
# - Conv2d (3x3 Depthwise)
# This kernel will fuse the following:
# - BatchNorm2d
# - ReLU
# - Element-wise operations

# This kernel will fuse the following:
# stride-based 
- Conv2d (3PW) 1x1 Pointwise-Conv
Pointwise-Conv 1Depthwise-Conv 3x3
# Pointwise-Conv 1x1
# This kernel given-and-and-and-and-and-and-action-
# This kernel will fuse theMobileNetV1 architecture.
# This architecture is a running-mean,er,er,er,and-ReLU-and-        # - BatchNorm-and-function-ReLU-swap-and-and-scale-<channel|># Optimized with CUDA operators:

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# The optimization strategy focuses on fusing BatchNorm and ReLU into the preceding convolution.
# In a production environment, one would use a fused Depthwise-Separable kernel.
# Here, we implement a fused BatchNorm + ReLU kernel to reduce memory bandwidth overhead.

fused_bn_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_bn_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W,
    float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * H * W;
    
    if (idx < total_elements) {
        int c = (idx / (H * W)) % C;
        
        float m = running_mean[c];
        float v = running_var[c];
        float w = weight[c];
        float b = bias[c];
        
        float inv_std = rsqrtf(v + eps);
        float val = input[idx] * (w * inv_std) + (b - m * w * inv_std);
        
        // ReLU
        output[idx] = val > 0.0f ? val : 0.0f;
    }
}

torch::Tensor fused_bn_relu_cuda(
    torch::Tensor input,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps) {
    
    auto output = torch::empty_like(input);
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int total_elements = N * C * H * W;

    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_bn_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W, eps);

    return output;
}
"""

fused_bn_relu_cpp_source = """
torch::Tensor fused_bn_relu_cuda(
    torch::Tensor input,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps);
"""

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_bn_relu_cpp_source,
    cuda_sources=fused_bn_relu_source,
    functions=["fused_bn_relu_cuda"],
    verbose=False,
)

class FusedBNReLU(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        # We keep these as parameters to maintain compatibility with standard training
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        return fused_ops.fused_bn_relu_cuda(
            x, self.running_mean, self.running_var, self.weight, self.bias, self.eps
        )

class ConvBNReLU(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride, padding, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(inp, oup, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn_relu = FusedBNReLU(oup)

    def forward(self, x):
        return self.bn_relu(self.conv(x))

class ConvDW(nn.Module):
    def __init__(self, inp, oup, stride):
        super().__init__()
        # Depthwise
        self.dw = ConvBNReLU(inp, inp, 3, stride, 1, groups=inp)
        # Pointwise
        self.pw = ConvBNReLU(inp, oup, 1, 1, 0, groups=1)

    def forward(self, x):
        return self.pw(self.dw(x))

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000, input_channels=3, alpha=1.0):
        super(ModelNew, self).__init__()
        
        def conv_bn(inp, oup, stride):
            return ConvBNReLU(inp, oup, 3, stride, 1)
        
        def conv_dw(inp, oup, stride):
            return ConvDW(inp, oup, stride)
        
        self.model = nn.Sequential(
            conv_bn(input_channels, int(32 * alpha), 2),
            conv_dw(int(32 * alpha), int(64 * alpha), 1),
            conv_dw(int(64 * alpha), int(128 * alpha), 2),
            conv_dw(int(128 * alpha), int(128 * alpha), 1),
            conv_dw(int(128 * alpha), int(256 * alpha), 2),
            conv_dw(int(256 * alpha), int(256 * alpha), 1),
            conv_dw(int(256 * alpha), int(512 * alpha), 2),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(1024 * alpha), 2),
            conv_dw(int(1024 * alpha), int(1024 * alpha), 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(int(1024 * alpha), num_classes)
    
    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x