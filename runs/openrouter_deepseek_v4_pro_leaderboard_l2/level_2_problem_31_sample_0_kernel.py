import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused post-convolution operations
fused_post_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_post_conv_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float constant_value,
    const float* __restrict__ bias,
    float scaling_factor,
    int total_elements,
    int spatial_size,
    int channels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float val = input[idx];
        val = fminf(val, constant_value);
        int c = (idx / spatial_size) % channels;
        val += bias[c];
        val *= scaling_factor;
        output[idx] = val;
    }
}

torch::Tensor fused_post_conv_cuda(
    torch::Tensor input,
    float constant_value,
    torch::Tensor bias,
    float scaling_factor
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");

    auto output = torch::empty_like(input);
    int total_elements = input.numel();
    int spatial_size = input.size(2) * input.size(3);
    int channels = input.size(1);

    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_post_conv_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        constant_value,
        bias.data_ptr<float>(),
        scaling_factor,
        total_elements,
        spatial_size,
        channels
    );

    return output;
}
"""

fused_post_conv_cpp_source = (
    "torch::Tensor fused_post_conv_cuda(torch::Tensor input, float constant_value, torch::Tensor bias, float scaling_factor);"
)

# Compile the inline CUDA code
fused_post_conv = load_inline(
    name="fused_post_conv",
    cpp_sources=fused_post_conv_cpp_source,
    cuda_sources=fused_post_conv_source,
    functions=["fused_post_conv_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernel fusing min, bias add, and scale after convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.fused_post_conv = fused_post_conv

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_post_conv.fused_post_conv_cuda(
            x, self.constant_value, self.bias, self.scaling_factor
        )
        return x