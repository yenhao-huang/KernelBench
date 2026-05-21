import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused Mish + Add + Hardtanh + Scale
fused_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_activation_kernel(
    const float* input, float* output, int size,
    float add_value, float min_val, float max_val, float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        // Mish activation: x * tanh(softplus(x))
        float sp = logf(1.0f + expf(x));
        float tanh_sp = tanhf(sp);
        float mish = x * tanh_sp;
        // Add
        float added = mish + add_value;
        // Hardtanh
        float clamped = fminf(fmaxf(added, min_val), max_val);
        // Scale
        output[idx] = clamped * scale;
    }
}

torch::Tensor fused_activation_cuda(
    torch::Tensor input,
    float add_value,
    float min_val,
    float max_val,
    float scale
) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_activation_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size,
        add_value,
        min_val,
        max_val,
        scale
    );

    return output;
}
"""

fused_activation_cpp_source = (
    "torch::Tensor fused_activation_cuda(torch::Tensor input, float add_value, float min_val, float max_val, float scale);"
)

# Compile the inline CUDA code
fused_activation = load_inline(
    name="fused_activation",
    cpp_sources=fused_activation_cpp_source,
    cuda_sources=fused_activation_source,
    functions=["fused_activation_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, then applies a fused
    Mish + Add + Hardtanh + Scale operation using a custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.add_value = add_value
        self.scale = scale
        self.fused_activation = fused_activation

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_activation.fused_activation_cuda(x, self.add_value, -1.0, 1.0, self.scale)
        return x