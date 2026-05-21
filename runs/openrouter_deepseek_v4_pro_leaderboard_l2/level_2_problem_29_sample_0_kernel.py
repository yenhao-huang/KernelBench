import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for linear + mish + mish fusion
fused_linear_mish_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ float mish_activation(float x) {
    float softplus = logf(1.0f + expf(x));
    return x * tanhf(softplus);
}

__global__ void fused_linear_mish_mish_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_features, int out_features) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch_size) return;
    
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= out_features) return;
    
    float sum = bias[col];
    for (int k = 0; k < in_features; ++k) {
        sum += input[row * in_features + k] * weight[col * in_features + k];
    }
    
    // First Mish
    float mish1 = mish_activation(sum);
    // Second Mish
    float mish2 = mish_activation(mish1);
    
    output[row * out_features + col] = mish2;
}

torch::Tensor fused_linear_mish_mish_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int batch_size, int in_features, int out_features) {
    
    auto output = torch::empty({batch_size, out_features}, input.options());
    
    dim3 block_size(16, 16);
    dim3 grid_size(
        (batch_size + block_size.x - 1) / block_size.x,
        (out_features + block_size.y - 1) / block_size.y
    );
    
    fused_linear_mish_mish_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, in_features, out_features
    );
    
    return output;
}
"""

fused_linear_mish_mish_cpp_source = (
    "torch::Tensor fused_linear_mish_mish_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias,"
    "int batch_size, int in_features, int out_features);"
)

# Compile the inline CUDA code
fused_linear_mish_mish = load_inline(
    name="fused_linear_mish_mish",
    cpp_sources=fused_linear_mish_mish_cpp_source,
    cuda_sources=fused_linear_mish_mish_source,
    functions=["fused_linear_mish_mish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.fused_linear_mish_mish = fused_linear_mish_mish

    def forward(self, x):
        return self.fused_linear_mish_mish.fused_linear_mish_mish_cuda(
            x, self.linear.weight, self.linear.bias,
            x.size(0), self.linear.in_features, self.linear.out_features
        )