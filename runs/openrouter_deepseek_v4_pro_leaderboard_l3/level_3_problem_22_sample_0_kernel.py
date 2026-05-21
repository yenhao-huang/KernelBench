```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA source code for fused BatchNorm + activation (ReLU/ReLU6)
fused_bn_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_bn_activation_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float eps,
    const int N,
    const int C,
    const int H,
    const int W,
    const float min_val,
    const float max_val,
    const bool has_activation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int c = (idx / (W * H)) % C;
    int n = idx / (W * H * C);

    float x = input[idx];
    int c_idx = c;
    float m = mean[c_idx];
    float v = var[c_idx];
    float wgt = weight[c_idx];
    float b = bias[c_idx];

    float y = ((x - m) * rsqrtf(v + eps)) * wgt + b;

    if (has_activation) {
        y = fminf(fmaxf(y, min_val), max_val);
    }

    output[idx] = y;
}

torch::Tensor fused_bn_activation_cuda(
    torch::Tensor input,
    torch::Tensor mean,
    torch::Tensor var,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps,
    const std::string& activation_type
) {
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(mean.is_cuda(), "Mean must be on CUDA");
    TORCH_CHECK(var.is_cuda(), "Var must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight must be on CUDA");
    TORCH_CHECK(bias.is_cuda(), "Bias must be on CUDA");

    auto N = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);
    int total = N * C * H * W;

    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;

    float min_val, max_val;
    bool has_activation = false;
    if (activation_type == "relu") {
        min_val = 0.0f;
        max_val = INFINITY;
        has_activation = true;
    } else if (activation_type == "relu6") {
        min_val = 0.0f;
        max_val = 6.0f;
        has_activation = true;
    }

    fused_bn_activation_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        static_cast<float>(eps),
        N, C, H, W,
        min_val, max_val, has_activation
    );

    return output;
}
"""

fused_bn_cpp_source = (
    "torch::Tensor fused_bn_activation_cuda(torch::Tensor input, torch::Tensor mean, "
    "torch::Tensor var, torch::Tensor weight, torch::Tensor bias, double eps, "
    "const std::string& activation_type);"
)

# Compile the inline CUDA module
fused_bn_ops = load_inline(
    name="fused_bn_ops",
    cpp_sources=fused_bn_cpp_source,
    cuda_sources=fused_bn_activation_source,
    functions=["fused_bn_activation_cuda"],
    verbose=False,
    extra_cflags=["-O2"],
    extra_ldflags=[],
)


# Custom autograd functions for fused BN + ReLU / ReLU6 (inference only)
class FusedBNReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mean, var, weight, bias, eps):
        output = fused_bn_ops.fused_bn_activation_cuda(
            input, mean.contiguous(), var.contiguous(),
            weight.contiguous(), bias.contiguous(), eps, "relu"
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Training not supported; all gradients are None
        return None, None, None, None, None, None


class FusedBNReLU6(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mean, var, weight, bias, eps):
        output = fused_bn_ops.fused_bn_activation_cuda(
            input, mean.contiguous(), var.contiguous(),
            weight.contiguous(), bias.contiguous(), eps, "relu6"
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None, None, None


class MBConvNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(MBConvNew, self).__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio
        self.expand_ratio = expand_ratio

        if expand_ratio != 1:
            self.expand_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1,
                                         stride=1, padding=0, bias=False)
            self.expand_bn = nn.BatchNorm2d(hidden_dim)

        self.depthwise_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size,
                                        stride=stride, padding=(kernel_size-1)//2,
                                        groups=hidden_dim, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(hidden_dim)

        self.project_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1,
                                      stride=1, padding=0, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x

        if self.expand_ratio != 1:
            x = self.expand_conv(x)
            x = FusedBNReLU6.apply(x, self.expand_bn.running_mean,
                                   self.expand_bn.running_var,
                                   self.expand_bn.weight,
                                   self.expand_bn.bias,
                                   self.expand_bn.eps)

        x = self.depthwise_conv(x)
        x = FusedBNReLU6.apply(x, self.depthwise_bn.running_mean,
                               self.depthwise_bn.running_var,
                               self.depthwise_bn.weight,
                               self.depthwise_bn.bias,
                               self.depthwise_bn.eps)

        x = self.project_conv(x)
        x = self.project_bn(x)  # No activation after projection

        if self.use_residual:
            x += identity

        return x


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.blocks = nn.Sequential(
            MBConvNew(32, 16, kernel_size=3, stride=1, expand_ratio=1),
            MBConvNew(16, 24,