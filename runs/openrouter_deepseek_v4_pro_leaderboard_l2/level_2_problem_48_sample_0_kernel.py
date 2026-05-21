import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source code for fused forward and backward kernels
fused_tanh_sigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_forward_kernel(const float* x, const float* scaling, const float* bias,
                                     float* y, float* b,
                                     int N, int C, int D, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * D * H * W;
    if (idx < total) {
        int w = idx % W;
        int h = (idx / W) % H;
        int d = (idx / (W * H)) % D;
        int c = (idx / (W * H * D)) % C;
        float val = x[idx];
        val = val * scaling[c];
        float tanh_val = tanhf(val);
        b[idx] = tanh_val;
        val = tanh_val * bias[c];
        y[idx] = 1.0f / (1.0f + expf(-val));
    }
}

__global__ void fused_backward_kernel(const float* grad_y, const float* x, const float* b, const float* scaling, const float* bias,
                                      float* grad_x, float* grad_scaling, float* grad_bias,
                                      int N, int C, int D, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * D * H * W;
    if (idx < total) {
        int c = (idx / (W * H * D)) % C;
        float grad_y_val = grad_y[idx];
        float b_val = b[idx];
        float bias_val = bias[c];
        float scaling_val = scaling[c];
        float c_val = b_val * bias_val;
        float y_val = 1.0f / (1.0f + expf(-c_val));
        float dL_dc = grad_y_val * y_val * (1.0f - y_val);
        float dL_dbias = dL_dc * b_val;
        float dL_db = dL_dc * bias_val;
        float dL_da = dL_db * (1.0f - b_val * b_val);
        float dL_dx = dL_da * scaling_val;
        float x_val = x[idx];
        float dL_dscaling = dL_da * x_val;
        grad_x[idx] = dL_dx;
        atomicAdd(&grad_scaling[c], dL_dscaling);
        atomicAdd(&grad_bias[c], dL_dbias);
    }
}

torch::Tensor fused_forward_cuda(torch::Tensor x, torch::Tensor scaling, torch::Tensor bias) {
    auto N = x.size(0);
    auto C = x.size(1);
    auto D = x.size(2);
    auto H = x.size(3);
    auto W = x.size(4);
    auto y = torch::empty_like(x);
    auto b = torch::empty_like(x);
    int total = N * C * D * H * W;
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;
    fused_forward_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), scaling.data_ptr<float>(), bias.data_ptr<float>(),
        y.data_ptr<float>(), b.data_ptr<float>(),
        N, C, D, H, W);
    return torch::tuple({y, b});
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fused_backward_cuda(
    torch::Tensor grad_y, torch::Tensor x, torch::Tensor b, torch::Tensor scaling, torch::Tensor bias) {
    auto N = x.size(0);
    auto C = x.size(1);
    auto D = x.size(2);
    auto H = x.size(3);
    auto W = x.size(4);
    auto grad_x = torch::empty_like(x);
    auto grad_scaling = torch::zeros_like(scaling);
    auto grad_bias = torch::zeros_like(bias);
    int total = N * C * D * H * W;
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;
    fused_backward_kernel<<<num_blocks, block_size>>>(
        grad_y.data_ptr<float>(), x.data_ptr<float>(), b.data_ptr<float>(),
        scaling.data_ptr<float>(), bias.data_ptr<float>(),
        grad_x.data_ptr<float>(), grad_scaling.data_ptr<float>(), grad_bias.data_ptr<float>(),
        N, C, D, H, W);
    return std::make_tuple(grad_x, grad_scaling, grad_bias);
}
"""

fused_tanh_sigmoid_cpp_source = """
torch::Tensor fused_forward_cuda(torch::Tensor x, torch::Tensor scaling, torch::Tensor bias);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fused_backward_cuda(
    torch::Tensor grad_y, torch::Tensor x, torch::Tensor b, torch::Tensor scaling, torch::Tensor bias);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_tanh_sigmoid",
    cpp_sources=fused_tanh_sigmoid_cpp_source,
    cuda_sources=fused_tanh_sigmoid_source,
    functions=["fused_forward_cuda", "fused_backward_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Custom autograd function wrapping the fused kernels
class FusedTanhSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scaling, bias):
        y, b = fused_ops.fused_forward_cuda(x, scaling, bias)
        ctx.save_for_backward(x, b, scaling, bias)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, b, scaling, bias = ctx.saved_tensors
        grad_x, grad_scaling, grad_bias = fused_ops.fused_backward_cuda(grad_y, x, b, scaling, bias)
        return grad_x, grad_scaling, grad_bias


class ModelNew(nn.Module):
    """
    Optimized model with fused tanh-sigmoid CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv(x)
        x = FusedTanhSigmoid.apply(x, self.scaling_factor, self.bias)
        return x