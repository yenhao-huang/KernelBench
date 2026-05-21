import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for fused bias subtraction and tanh
fused_bias_sub_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void bias_sub_tanh_kernel(float* out, const float* in, const float* bias,
                                     int N, int C, int H, int W) {
    int total = N * C * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        // Compute indices
        int w = idx % W;
        int h = (idx / W) % H;
        int c = (idx / (W * H)) % C;
        // int n = idx / (C * H * W); // not needed for bias indexing
        out[idx] = tanhf(in[idx] - bias[c]);
    }
}

torch::Tensor fused_bias_sub_tanh_cuda(torch::Tensor input, torch::Tensor bias) {
    // input shape: (N, C, H, W), bias shape: (C, 1, 1)
    auto N = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);
    auto output = torch::empty_like(input);

    int total = N * C * H * W;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    bias_sub_tanh_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        N, C, H, W
    );

    return output;
}
"""

fused_bias_sub_tanh_cpp_source = """
torch::Tensor fused_bias_sub_tanh_cuda(torch::Tensor input, torch::Tensor bias);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_bias_sub_tanh",
    cpp_sources=fused_bias_sub_tanh_cpp_source,
    cuda_sources=fused_bias_sub_tanh_source,
    functions=["fused_bias_sub_tanh_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model: transposed convolution, then fused bias subtraction + tanh.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_op.fused_bias_sub_tanh_cuda(x, self.bias)
        return x


# The following functions are kept for compatibility with the original interface
batch_size = 32
in_channels = 64
out_channels = 64
height = width = 256
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]