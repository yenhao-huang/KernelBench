import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA source for fused BatchNorm + Activation
fused_bn_act_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_bn_act_kernel(float* out, const float* in, const float* weight, const float* bias, const float* mean, const float* var, float eps, int act_type, int N, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx < total) {
        int w = idx % W;
        int h = (idx / W) % H;
        int c = (idx / (W * H)) % C;
        int n = idx / (W * H * C);
        float val = in[idx];
        float mean_c = mean[c];
        float var_c = var[c];
        float inv_std = rsqrtf(var_c + eps);
        float w_c = weight[c];
        float b_c = bias[c];
        float normalized = (val - mean_c) * inv_std * w_c + b_c;
        if (act_type == 0) {
            out[idx] = fmaxf(normalized, 0.0f);
        } else if (act_type == 1) {
            out[idx] = fminf(fmaxf(normalized, 0.0f), 6.0f);
        } else {
            out[idx] = normalized;
        }
    }
}

torch::Tensor fused_bn_act_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor running_mean, torch::Tensor running_var, double eps, int act_type) {
    auto N = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);
    auto output = torch::empty_like(input);
    const int block_size = 256;
    const int num_blocks = (N * C * H * W + block_size - 1) / block_size;
    fused_bn_act_kernel<<<num_blocks, block_size>>>(output.data_ptr<float>(), input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), running_mean.data_ptr<float>(), running_var.data_ptr<float>(), (float)eps, act_type, N, C, H, W);
    return output;
}
"""

fused_bn_act_cpp_source = "torch::Tensor fused_bn_act_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor running_mean, torch::Tensor running_var, double eps, int act_type);"

# Compile the inline CUDA code
fused_bn_act_module = load_inline(
    name="fused_bn_act",
    cpp_sources=fused_bn_act_cpp_source,
    cuda_sources=fused_bn_act_source,
    functions=["fused_bn_act_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class FusedBNReLU(nn.Module):
    """Fused BatchNorm2d + ReLU"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.act_type = 0  # ReLU

    def forward(self, x):
        return fused_bn_act_module.fused_bn_act_cuda(
            x, self.weight, self.bias, self.running_mean, self.running_var, self.eps, self.act_type
        )

class FusedBNReLU6(nn.Module):
    """Fused BatchNorm2d + ReLU6"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.act_type = 1  # ReLU6

    def forward(self, x):
        return fused_bn_act_module.fused_bn_act_cuda(
            x, self.weight, self.bias, self.running_mean, self.running_var, self.eps, self.act_type
        )

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = FusedBNReLU(32)  # fused BN + ReLU
        
        # MBConv blocks
        self.mbconv1 = self._make_mbconv_block(32, 16, 1, 1)
        self.mbconv2 = self._make_mbconv_block(16, 24, 2, 6)
        self.mbconv3 = self._make_mbconv_block(24, 40, 2, 6)
        self.mbconv4 = self._make_mbconv_block(40, 80, 2, 6)
        self.mbconv5 = self._make_mbconv_block(80, 112, 1, 6)
        self.mbconv6 = self._make_mbconv_block(112, 192, 2, 6)
        self.mbconv7 = self._make_mbconv_block(192, 320, 1, 6)
        
        # Final convolutional layer
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = FusedBNReLU(1280)  # fused BN + ReLU
        
        # Fully connected layer
        self.fc = nn.Linear(1280, num_classes)
    
    def _make_mbconv_block(self, in_channels, out_channels, stride, expand_ratio):
        hidden_dim = round(in_channels * expand_ratio)
        return nn.Sequential(
            # 1x1 expansion conv + BN + ReLU6 (fused)
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            FusedBNReLU6(hidden_dim),
            # 3x3 depthwise conv + BN + ReLU6 (fused)
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            FusedBNReLU6(hidden_dim),
            # 1x1 projection conv + BN (no activation, keep original BN)
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, x):
        x = self.bn1(self.conv1(x))  # conv1 + fused BN+ReLU
        
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)
        x = self.mbconv6(x)
        x = self.mbconv7(x)
        
        x = self.bn2(self.conv2(x))  # conv2 + fused BN+ReLU
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x