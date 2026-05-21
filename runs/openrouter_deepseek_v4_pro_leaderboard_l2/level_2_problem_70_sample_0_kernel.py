import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for fused sigmoid, scaling, and residual addition
sigmoid_scale_residual_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void sigmoid_scale_residual_kernel(const float* input, float* output, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        output[idx] = 1.0f / (1.0f + expf(-val)) * scale + val;
    }
}

torch::Tensor sigmoid_scale_residual_cuda(torch::Tensor input, float scale) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    sigmoid_scale_residual_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), scale, size);

    return output;
}
"""

sigmoid_scale_residual_cpp_source = """
torch::Tensor sigmoid_scale_residual_cuda(torch::Tensor input, float scale);
"""

# Compile the inline CUDA code
sigmoid_scale_residual = load_inline(
    name="sigmoid_scale_residual",
    cpp_sources=sigmoid_scale_residual_cpp_source,
    cuda_sources=sigmoid_scale_residual_source,
    functions=["sigmoid_scale_residual_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(input_size, hidden_size)
        self.scaling_factor = scaling_factor
        self.sigmoid_scale_residual = sigmoid_scale_residual

    def forward(self, x):
        x = self.gemm(x)
        x = self.sigmoid_scale_residual.sigmoid_scale_residual_cuda(x, self.scaling_factor)
        return x