import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# CUDA source for fused linear + ReLU + division
fused_linear_relu_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_linear_relu_div_kernel(const float* x, const float* weight, const float* bias, float* out,
                                             int batch_size, int in_features, int out_features, float divisor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_features;
    if (idx < total) {
        int row = idx / out_features;
        int col = idx % out_features;
        float sum = bias[col];
        for (int k = 0; k < in_features; ++k) {
            sum += x[row * in_features + k] * weight[col * in_features + k];
        }
        sum = fmaxf(sum, 0.0f);
        out[idx] = sum / divisor;
    }
}

torch::Tensor fused_linear_relu_div_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, float divisor) {
    int batch_size = x.size(0);
    int in_features = x.size(1);
    int out_features = weight.size(0);
    auto out = torch::empty({batch_size, out_features}, x.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * out_features + block_size - 1) / block_size;

    fused_linear_relu_div_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(),
        batch_size, in_features, out_features, divisor
    );

    return out;
}
"""

fused_linear_relu_div_cpp_source = "torch::Tensor fused_linear_relu_div_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, float divisor);"

# Compile the inline CUDA code
fused_linear_relu_div = load_inline(
    name="fused_linear_relu_div",
    cpp_sources=fused_linear_relu_div_cpp_source,
    cuda_sources=fused_linear_relu_div_source,
    functions=["fused_linear_relu_div_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, divisor):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.divisor = divisor
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()
        self.fused_op = fused_linear_relu_div

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return self.fused_op.fused_linear_relu_div_cuda(x, self.weight, self.bias, self.divisor)