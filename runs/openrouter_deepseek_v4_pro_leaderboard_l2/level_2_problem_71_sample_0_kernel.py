import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused division and LeakyReLU
fused_div_leakyrelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_div_leakyrelu_kernel(const float* input, float* output, int size, float divisor, float negative_slope) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx] / divisor;
        output[idx] = (val > 0.0f) ? val : (val * negative_slope);
    }
}

torch::Tensor fused_div_leakyrelu_cuda(torch::Tensor input, float divisor, float negative_slope) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_div_leakyrelu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), size, divisor, negative_slope
    );

    return output;
}
"""

fused_div_leakyrelu_cpp_source = (
    "torch::Tensor fused_div_leakyrelu_cuda(torch::Tensor input, float divisor, float negative_slope);"
)

# Compile the inline CUDA code
fused_div_leakyrelu = load_inline(
    name="fused_div_leakyrelu",
    cpp_sources=fused_div_leakyrelu_cpp_source,
    cuda_sources=fused_div_leakyrelu_source,
    functions=["fused_div_leakyrelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA operator fusing division and LeakyReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.fused_op = fused_div_leakyrelu

    def forward(self, x):
        x = self.conv(x)
        # Fused division and LeakyReLU with negative_slope=0.01
        x = self.fused_op.fused_div_leakyrelu_cuda(x, self.divisor, 0.01)
        return x