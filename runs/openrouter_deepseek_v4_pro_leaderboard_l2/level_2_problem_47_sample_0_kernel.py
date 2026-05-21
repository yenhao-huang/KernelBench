import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Fused Mish + Tanh activation CUDA kernel
fused_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_mish_tanh_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        // Mish: x * tanh(softplus(x)) where softplus(x) = log(1 + exp(x))
        float sp = logf(1.0f + expf(x));
        float mish = x * tanhf(sp);
        // Tanh
        output[idx] = tanhf(mish);
    }
}

torch::Tensor fused_mish_tanh_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_mish_tanh_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

fused_activation_cpp_source = (
    "torch::Tensor fused_mish_tanh_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code
fused_activation = load_inline(
    name="fused_activation",
    cpp_sources=fused_activation_cpp_source,
    cuda_sources=fused_activation_source,
    functions=["fused_mish_tanh_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA fused Mish+Tanh activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.fused_activation = fused_activation

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_activation.fused_mish_tanh_cuda(x)
        return x