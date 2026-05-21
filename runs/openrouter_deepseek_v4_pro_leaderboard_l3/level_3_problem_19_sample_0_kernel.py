import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

# ------------------------------------------------------------
# CUDA code for fused batch norm + ReLU
# ------------------------------------------------------------
fused_bn_relu_cpp_source = "torch::Tensor fused_bn_relu_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, torch::Tensor running_mean, torch::Tensor running_var, float eps);"

fused_bn_relu_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_bn_relu_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    float eps,
    int N, int C, int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx < total) {
        int c = (idx / (W * H)) % C;
        float val = x[idx];
        float norm = (val - mean[c]) * rsqrtf(var[c] + eps);
        float scaled = norm * gamma[c] + beta[c];
        out[idx] = fmaxf(scaled, 0.0f);
    }
}

torch::Tensor fused_bn_relu_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps)
{
    auto N = x.size(0);
    auto C = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);
    auto out = torch::empty_like(x);

    int total = N * C * H * W;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    fused_bn_relu_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        eps, N, C, H, W);

    return out;
}
"""

fused_bn_relu_op = load_inline(
    name="fused_bn_relu",
    cpp_sources=fused_bn_relu_cpp_source,
    cuda_sources=fused_bn_relu_cuda_source,
    functions=["fused_bn_relu_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# ------------------------------------------------------------
# Custom fused modules replacing original blocks
# ------------------------------------------------------------
class ConvBNReLU(nn.Module):
    """Standard convolution + BN + ReLU fused at the BN/ReLU level."""
    def __init__(self, in_planes, out_planes, stride, padding=1):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.conv_weight = nn.Parameter(torch.empty(out_planes, in_planes, 3, 3))
        nn.init.kaiming_uniform_(self.conv_weight, a=math.sqrt(5))
        self.bn_weight = nn.Parameter(torch.ones(out_planes))
        self.bn_bias = nn.Parameter(torch.zeros(out_planes))
        self.register_buffer('running_mean', torch.zeros(out_planes))
        self.register_buffer('running_var', torch.ones(out_planes))
        self.eps = 1e-5

    def forward(self, x):
        x = F.conv2d(x, self.conv_weight, bias=None, stride=self.stride, padding=self.padding)
        x = fused_bn_relu_op.fused_bn_relu_cuda(
            x, self.bn_weight, self.bn_bias, self.running_mean, self.running_var, self.eps)
        return x


class DepthwiseConvBNReLU(nn.Module):
    """Depthwise convolution (groups=in_planes) + BN + ReLU fused at the BN/ReLU level."""
    def __init__(self, in_planes, stride, padding=1):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.groups = in_planes
        self.conv_weight = nn.Parameter(torch.empty(in_planes, 1, 3, 3))
        nn.init.kaiming_uniform_(self.conv_weight, a=math.sqrt(5))
        self.bn_weight = nn.Parameter(torch.ones(in_planes))
        self.bn_bias = nn.Parameter(torch.zeros(in_planes))
        self.register_buffer('running_mean', torch.zeros(in_planes))
        self.register_buffer('running_var', torch.ones(in_planes))
        self.eps = 1e-5

    def forward(self, x):
        x = F.conv2d(x, self.conv_weight, bias=None, stride=self.stride,
                     padding=self.padding, groups=self.groups)
        x = fused_bn_relu_op.fused_bn_relu_cuda(
            x, self.bn_weight, self.bn_bias, self.running_mean, self.running_var, self.eps)
        return x


class PointwiseConvBNReLU(nn.Module):
    """Pointwise (1x1) convolution + BN + ReLU fused at the BN/ReLU level."""
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.conv_weight = nn.Parameter(torch.empty(out_planes, in_planes, 1, 1))
        nn.init.kaiming_uniform_(self.conv_weight, a=math.sqrt(5))
        self.bn_weight = nn.Parameter(torch.ones(out_planes))
        self.bn_bias = nn.Parameter(torch.zeros(out_planes))
        self.register_buffer('running_mean', torch.zeros(out_planes))
        self.register_buffer('running_var', torch.ones(out_planes))
        self.eps = 1e-5

    def forward(self, x):
        x = F.conv2d(x, self.conv_weight, bias=None, stride=1, padding=0)
        x = fused_bn_relu_op.fused_bn_relu_cuda(
            x, self.bn_weight, self.bn_bias, self.running_mean, self.running_var, self.eps)
        return x


# ------------------------------------------------------------
# Optimized model
# ------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, num_classes=1000, input_channels=3, alpha=1.0):
        super().__init__()

        # Build the layer list exactly matching the original MobileNetV1 structure
        layers = []
        # First standard conv block
        layers.append(ConvBNReLU(input_channels, int(32 * alpha), stride=2, padding=1))

        # conv_dw blocks are split into depthwise + pointwise sub-blocks
        cfg = [
            # (in_dw, out_dw, out_pw, stride)
            (32, 32, 64, 1),     # dw stride 1, pw 32->64
            (64, 64, 128, 2),    # dw stride 2, pw 64->128
            (128, 128, 128, 1),  # dw stride 1, pw 128->128
            (128, 128, 256, 2),  # dw stride 2, pw 128->256
            (256, 256, 256, 1),  # dw stride 1, pw 256->256
            (256, 256, 512, 2),  # dw stride 2, pw 256->512
            (512, 512, 512, 1),  # repeat x5
            (512, 512, 512, 1),
            (512, 512, 512, 1),
            (512, 512, 512, 1),
            (512, 512, 512, 1),
            (512, 512, 1024, 2), # dw stride 2, pw 512->1024
            (1024, 1024, 1024, 1) # final
        ]

        for in_dw, out_dw, out_pw, stride in cfg:
            # Apply alpha multiplier
            in_dw = int(in_dw * alpha)
            out_dw = int(out_dw * alpha)
            out_pw = int(out_pw * alpha)

            layers.append(DepthwiseConvBNReLU(in_dw, stride=stride, padding=1))
            layers.append(PointwiseConvBNReLU(out_dw, out_pw))

        # Final average pooling and fully connected
        layers.append(nn.AvgPool2d(7))
        self.layers = nn.ModuleList(layers)
        self.fc = nn.Linear(int(1024 * alpha), num_classes)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x