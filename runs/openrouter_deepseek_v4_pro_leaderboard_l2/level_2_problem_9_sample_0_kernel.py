import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused linear + subtract + multiply + relu
fused_linear_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_linear_ops_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_features, int out_features,
    float subtract_value, float multiply_value) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch_size) return;
    
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= out_features) return;
    
    float sum = 0.0f;
    for (int k = 0; k < in_features; ++k) {
        sum += input[row * in_features + k] * weight[col * in_features + k];
    }
    sum += bias[col];
    sum = sum - subtract_value;
    sum = sum * multiply_value;
    output[row * out_features + col] = fmaxf(sum, 0.0f);
}

torch::Tensor fused_linear_ops_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float subtract_value, float multiply_value) {
    
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::empty({batch_size, out_features}, input.options());
    
    const int block_size_x = 16;
    const int block_size_y = 16;
    dim3 block_size(block_size_x, block_size_y);
    dim3 num_blocks(
        (batch_size + block_size_x - 1) / block_size_x,
        (out_features + block_size_y - 1) / block_size_y
    );
    
    fused_linear_ops_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, in_features, out_features,
        subtract_value, multiply_value
    );
    
    return output;
}
"""

fused_linear_ops_cpp_source = (
    "torch::Tensor fused_linear_ops_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias,"
    "float subtract_value, float multiply_value);"
)

# Compile the inline CUDA code
fused_linear_ops = load_inline(
    name="fused_linear_ops",
    cpp_sources=fused_linear_ops_cpp_source,
    cuda_sources=fused_linear_ops_source,
    functions=["fused_linear_ops_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value
        self.fused_linear_ops = fused_linear_ops

    def forward(self, x):
        return self.fused_linear_ops.fused_linear_ops_cuda(
            x, self.linear.weight, self.linear.bias,
            self.subtract_value, self.multiply_value
        )