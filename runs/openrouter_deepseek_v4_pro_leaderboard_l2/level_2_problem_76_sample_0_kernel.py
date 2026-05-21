import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused bias addition and ReLU
bias_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void bias_relu_kernel(const float* input, const float* bias, float* output, int num_elements, int num_cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        int col = idx % num_cols;
        float val = input[idx] + bias[col];
        output[idx] = val > 0.0f ? val : 0.0f;
    }
}

torch::Tensor bias_relu_cuda(torch::Tensor input, torch::Tensor bias) {
    auto num_elements = input.numel();
    auto output = torch::zeros_like(input);
    int num_cols = input.size(1);

    const int block_size = 256;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    bias_relu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), num_elements, num_cols);

    return output;
}
"""

bias_relu_cpp_source = "torch::Tensor bias_relu_cuda(torch::Tensor input, torch::Tensor bias);"

# Compile the inline CUDA code for fused bias+ReLU
bias_relu = load_inline(
    name="bias_relu",
    cpp_sources=bias_relu_cpp_source,
    cuda_sources=bias_relu_source,
    functions=["bias_relu_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.bias_relu = bias_relu

    def forward(self, x):
        x = self.gemm(x)
        x = self.bias_relu.bias_relu_cuda(x, self.bias)
        return x