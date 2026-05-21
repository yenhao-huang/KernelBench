import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for fused multiply + LeakyReLU + GELU
fused_ops_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define NEG_SLOPE 0.01f

__global__ void fused_multiply_leakyrelu_gelu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ multiplier,
    float* __restrict__ output,
    int N, int C, int H, int W)
{
    int total_elements = N * C * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // Compute n, c, h, w from flat index
    int w = idx % W;
    int h = (idx / W) % H;
    int c = (idx / (W * H)) % C;
    int n = idx / (W * H * C);

    // Load input value
    float val = input[idx];

    // Multiply by channel-wise multiplier (broadcasted)
    val = val * multiplier[c];

    // LeakyReLU
    val = val > 0.0f ? val : val * NEG_SLOPE;

    // GELU approximation (tanh)
    float x3 = val * val * val;
    float inner = sqrtf(2.0f / M_PI) * (val + 0.044715f * x3);
    val = 0.5f * val * (1.0f + tanhf(inner));

    output[idx] = val;
}

torch::Tensor fused_multiply_leakyrelu_gelu_cuda(
    torch::Tensor input,
    torch::Tensor multiplier)
{
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(multiplier.is_cuda(), "multiplier must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");
    TORCH_CHECK(multiplier.dim() == 3 && multiplier.size(1) == 1 && multiplier.size(2) == 1,
                "multiplier must be 3D (C, 1, 1)");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int total_elements = N * C * H * W;

    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_multiply_leakyrelu_gelu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        multiplier.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W);

    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_multiply_leakyrelu_gelu_cuda(torch::Tensor input, torch::Tensor multiplier);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_cuda_source,
    functions=["fused_multiply_leakyrelu_gelu_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Model that performs a convolution, then a fused multiply + LeakyReLU + GELU using custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_ops.fused_multiply_leakyrelu_gelu_cuda(x, self.multiplier)
        return x