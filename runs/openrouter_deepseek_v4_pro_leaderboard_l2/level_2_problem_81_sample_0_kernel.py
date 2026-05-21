import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel source for the fused element-wise operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_swish_div_clamp_tanh_clamp_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        // Swish: x * sigmoid(x)
        float sig = 1.0f / (1.0f + expf(-x));
        float swish = x * sig;
        // Divide by 2.0
        float div = swish * 0.5f;
        // Clamp to [-1.0, 1.0]
        float clamped1 = fminf(fmaxf(div, -1.0f), 1.0f);
        // Tanh
        float tanh_val = tanhf(clamped1);
        // Clamp again to [-1.0, 1.0]
        float out = fminf(fmaxf(tanh_val, -1.0f), 1.0f);
        output[idx] = out;
    }
}

torch::Tensor fused_swish_div_clamp_tanh_clamp_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_swish_div_clamp_tanh_clamp_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_swish_div_clamp_tanh_clamp_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_swish_div_clamp_tanh_clamp",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_swish_div_clamp_tanh_clamp_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model with a custom CUDA kernel fusing swish, divide, clamp, tanh, and clamp.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.gemm(x)
        x = self.fused_ops.fused_swish_div_clamp_tanh_clamp_cuda(x)
        return x