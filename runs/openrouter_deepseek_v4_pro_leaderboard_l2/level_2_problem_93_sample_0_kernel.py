import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused add, min(0), GELU, multiply
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ float gelu_approx(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__global__ void fused_add_min_gelu_mul_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size,
    float add_value,
    float multiply_value
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx] + add_value;
        // min(x, 0.0f)
        if (x > 0.0f) x = 0.0f;
        // GELU activation
        x = gelu_approx(x);
        // multiply
        output[idx] = x * multiply_value;
    }
}

torch::Tensor fused_add_min_gelu_mul_cuda(
    torch::Tensor input,
    float add_value,
    float multiply_value
) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_add_min_gelu_mul_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size,
        add_value,
        multiply_value
    );

    return output;
}
"""

fused_kernel_cpp_source = (
    "torch::Tensor fused_add_min_gelu_mul_cuda(torch::Tensor input, float add_value, float multiply_value);"
)

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_add_min_gelu_mul",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_add_min_gelu_mul_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernel fusing add, min(0), GELU, and multiply.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_op.fused_add_min_gelu_mul_cuda(x, self.add_value, self.multiply_value)
        return x