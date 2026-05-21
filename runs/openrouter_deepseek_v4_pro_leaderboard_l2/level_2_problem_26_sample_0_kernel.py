import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused addition and HardSwish activation
fused_add_hardswish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_add_hardswish_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sum = a[idx] + b[idx];
        float val = sum + 3.0f;
        float relu6 = fminf(fmaxf(val, 0.0f), 6.0f);
        out[idx] = sum * relu6 / 6.0f;
    }
}

torch::Tensor fused_add_hardswish_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.sizes() == b.sizes());
    TORCH_CHECK(a.is_cuda() && b.is_cuda());
    TORCH_CHECK(a.dtype() == torch::kFloat32 && b.dtype() == torch::kFloat32);
    auto size = a.numel();
    auto out = torch::empty_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_add_hardswish_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

fused_add_hardswish_cpp_source = (
    "torch::Tensor fused_add_hardswish_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code
fused_add_hardswish_module = load_inline(
    name="fused_add_hardswish",
    cpp_sources=fused_add_hardswish_cpp_source,
    cuda_sources=fused_add_hardswish_source,
    functions=["fused_add_hardswish_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernel fusing addition and HardSwish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        # bias_shape is preserved but not used in forward to match original behavior
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_add_hardswish = fused_add_hardswish_module

    def forward(self, x, add_input):
        x = self.conv_transpose(x)
        x = self.fused_add_hardswish.fused_add_hardswish_cuda(x, add_input)
        return x