import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source code for the fused activation kernel
fused_activation_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_activation_kernel(
    const float* __restrict__ input,
    const float* __restrict__ sum,
    float* __restrict__ output,
    int N, int C, int D, int H, int W,
    float negative_slope, float clamp_min, float clamp_max)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * D * H * W;
    if (idx >= total) return;

    float val = input[idx];

    // LeakyReLU
    val = val > 0.0f ? val : val * negative_slope;

    // Add channel-wise bias (sum_tensor broadcasted)
    int spatial_size = D * H * W;
    int c = (idx / spatial_size) % C;
    val += sum[c];

    // Clamp
    val = fminf(fmaxf(val, clamp_min), clamp_max);

    // GELU activation (exact)
    const float sqrt2 = 1.41421356237f;
    val = 0.5f * val * (1.0f + erff(val / sqrt2));

    output[idx] = val;
}

torch::Tensor fused_activation_cuda(
    torch::Tensor input,
    torch::Tensor sum_tensor,
    float negative_slope,
    float clamp_min,
    float clamp_max)
{
    input = input.contiguous();
    auto N = input.size(0);
    auto C = input.size(1);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);
    int total = N * C * D * H * W;

    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;

    fused_activation_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        sum_tensor.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D, H, W,
        negative_slope, clamp_min, clamp_max);

    return output;
}
"""

fused_activation_cpp_source = "torch::Tensor fused_activation_cuda(torch::Tensor input, torch::Tensor sum_tensor, float negative_slope, float clamp_min, float clamp_max);"

# Compile the inline CUDA code
fused_activation = load_inline(
    name="fused_activation",
    cpp_sources=fused_activation_cpp_source,
    cuda_sources=fused_activation_cuda_source,
    functions=["fused_activation_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model using a fused CUDA kernel for LeakyReLU + add + clamp + GELU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))
        self.fused_activation = fused_activation

    def forward(self, x):
        x = self.conv(x)
        # Flatten sum_tensor to 1D for the kernel (shape: out_channels)
        sum_1d = self.sum_tensor.view(-1).contiguous()
        x = self.fused_activation.fused_activation_cuda(x, sum_1d, 0.2, -1.0, 1.0)
        return x