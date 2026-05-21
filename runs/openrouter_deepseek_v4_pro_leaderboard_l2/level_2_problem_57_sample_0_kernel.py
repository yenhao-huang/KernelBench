import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused ReLU + HardSwish
fused_relu_hardswish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_relu_hardswish_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float y = fmaxf(0.0f, x);
        output[idx] = y * fminf((y + 3.0f) / 6.0f, 1.0f);
    }
}

torch::Tensor fused_relu_hardswish_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_relu_hardswish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), size
    );

    return output;
}
"""

fused_relu_hardswish_cpp_source = (
    "torch::Tensor fused_relu_hardswish_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code
fused_relu_hardswish = load_inline(
    name="fused_relu_hardswish",
    cpp_sources=fused_relu_hardswish_cpp_source,
    cuda_sources=fused_relu_hardswish_source,
    functions=["fused_relu_hardswish_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.fused_act = fused_relu_hardswish

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_act.fused_relu_hardswish_cuda(x)
        return x