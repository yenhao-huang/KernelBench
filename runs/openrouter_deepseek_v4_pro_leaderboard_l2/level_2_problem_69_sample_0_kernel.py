import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused HardSwish + ReLU
fused_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_hardswish_relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        // HardSwish: x * min(max(x+3, 0), 6) / 6
        float h = x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f;
        // ReLU: max(h, 0) (redundant but kept for architectural fidelity)
        output[idx] = fmaxf(h, 0.0f);
    }
}

torch::Tensor fused_hardswish_relu_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_hardswish_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

fused_activation_cpp_source = (
    "torch::Tensor fused_hardswish_relu_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code
fused_activation = load_inline(
    name="fused_activation",
    cpp_sources=fused_activation_cpp_source,
    cuda_sources=fused_activation_source,
    functions=["fused_hardswish_relu_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Model that performs a convolution, applies HardSwish, and then ReLU,
    with the activation functions fused into a single CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.fused_activation = fused_activation

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        x = self.conv(x)
        x = self.fused_activation.fused_hardswish_relu_cuda(x)
        return x


# The following inputs are kept for compatibility with the original interface
batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]