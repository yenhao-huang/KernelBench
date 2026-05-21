import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA source for fused bias addition, division, and GELU activation
fused_bias_div_gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_bias_div_gelu_kernel(const float* input, const float* bias, float divisor, float* output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        int row = idx / cols;
        int col = idx % cols;
        float val = input[idx] + bias[col];
        val = val / divisor;
        // GELU activation using tanh approximation
        float cdf = 0.5f * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (val + 0.044715f * val * val * val)));
        output[idx] = val * cdf;
    }
}

torch::Tensor fused_bias_div_gelu_cuda(torch::Tensor input, torch::Tensor bias, float divisor) {
    auto rows = input.size(0);
    auto cols = input.size(1);
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (rows * cols + block_size - 1) / block_size;

    fused_bias_div_gelu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), bias.data_ptr<float>(), divisor, output.data_ptr<float>(), rows, cols);

    return output;
}
"""

fused_bias_div_gelu_cpp_source = "torch::Tensor fused_bias_div_gelu_cuda(torch::Tensor input, torch::Tensor bias, float divisor);"

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_bias_div_gelu",
    cpp_sources=fused_bias_div_gelu_cpp_source,
    cuda_sources=fused_bias_div_gelu_source,
    functions=["fused_bias_div_gelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor
        self.fused_op = fused_op

    def forward(self, x):
        # Perform matrix multiplication (without bias) using PyTorch's matmul
        x = torch.matmul(x, self.linear.weight.t())
        # Apply fused bias addition, division, and GELU
        x = self.fused_op.fused_bias_div_gelu_cuda(x, self.linear.bias, self.divisor)
        return x