import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA source code for fused Conv2d+ReLU and Linear+ReLU
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel: fused Conv2d (3x3, pad=1, stride=1) + ReLU
__global__ void fused_conv2d_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int C_out, int H, int W) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C_out * H * W;
    if (tid >= total_elements) return;
    
    int w_out = tid % W;
    int h_out = (tid / W) % H;
    int c_out = (tid / (H * W)) % C_out;
    int n = tid / (H * W * C_out);
    
    float sum = bias[c_out];
    const int kernel_size = 3;
    const int pad = 1;
    
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int h_in = h_out + ky - pad;
                int w_in = w_out + kx - pad;
                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    float input_val = input[((n * C_in + c_in) * H + h_in) * W + w_in];
                    float weight_val = weight[((c_out * C_in + c_in) * kernel_size + ky) * kernel_size + kx];
                    sum += input_val * weight_val;
                }
            }
        }
    }
    output[((n * C_out + c_out) * H + h_out) * W + w_out] = fmaxf(sum, 0.0f);
}

// Kernel: fused Linear + ReLU
__global__ void fused_linear_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int in_features, int out_features) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * out_features;
    if (tid >= total_elements) return;
    
    int oc = tid % out_features;
    int n = tid / out_features;
    
    float sum = bias[oc];
    for (int ic = 0; ic < in_features; ++ic) {
        sum += input[n * in_features + ic] * weight[oc * in_features + ic];
    }
    output[n * out_features + oc] = fmaxf(sum, 0.0f);
}

// Wrapper functions callable from Python
torch::Tensor fused_conv2d_relu_cuda(
    torch::Tensor input,   // (N, C_in, H, W)
    torch::Tensor weight,  // (C_out, C_in, 3, 3)
    torch::Tensor bias     // (C_out)
) {
    TORCH_CHECK(input.dim() == 4, "Input must be 4D NCHW");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D");
    TORCH_CHECK(bias.dim() == 1, "Bias must be 1D");
    
    int N = input.size(0);
    int C_in = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int C_out = weight.size(0);
    
    auto output = torch::zeros({N, C_out, H, W}, input.options());
    
    int total_elements = N * C_out * H * W;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_conv2d_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, C_out, H, W
    );
    
    return output;
}

torch::Tensor fused_linear_relu_cuda(
    torch::Tensor input,   // (N, in_features)
    torch::Tensor weight,  // (out_features, in_features)
    torch::Tensor bias     // (out_features)
) {
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(weight.dim() == 2, "Weight must be 2D");
    TORCH_CHECK(bias.dim() == 1, "Bias must be 1D");
    
    int N = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::zeros({N, out_features}, input.options());
    
    int total_elements = N * out_features;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_linear_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, in_features, out_features
    );
    
    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_conv2d_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
torch::Tensor fused_linear_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_conv2d_relu_cuda", "fused_linear_relu_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class FusedConv2dReLU(nn.Module):
    """Conv2d (3x3, padding=1, stride=1) + ReLU fused"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, 3, 3) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        return fused_ops.fused_conv2d_relu_cuda(x, self.weight, self.bias)

class FusedLinearReLU(nn.Module):
    """Linear + ReLU fused"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        return fused_ops.fused_linear_relu_cuda(x, self.weight, self.bias)

import math

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        
        # Features: 5 blocks of Conv2d+ReLU pairs followed by MaxPooling
        self.features = nn.Sequential(
            # Block 1
            FusedConv2dReLU(3, 64),
            FusedConv2dReLU(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            FusedConv2dReLU(64, 128),
            FusedConv2dReLU(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            FusedConv2dReLU(128, 256),
            FusedConv2dReLU(256, 256),
            FusedConv2dReLU(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            FusedConv2dReLU(256, 512),
            FusedConv2dReLU(512, 512),
            FusedConv2dReLU(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5
            FusedConv2dReLU(512, 512),
            FusedConv2dReLU(512, 512),
            FusedConv2dReLU(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Classifier: fully-connected layers with fused Linear+ReLU and dropout
        self.classifier = nn.Sequential(
            FusedLinearReLU(512 * 7 * 7, 4096),
            nn.Dropout(p=0.0),
            FusedLinearReLU(4096, 4096),
            nn.Dropout(p=0.0),
            nn.Linear(4096, num_classes)  # last layer without ReLU
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x