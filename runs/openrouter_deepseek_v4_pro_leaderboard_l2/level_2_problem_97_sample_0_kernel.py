import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused bias addition, division, and Swish activation
fused_bias_div_swish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_bias_div_swish_kernel(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float div_value,
    float* __restrict__ out,
    int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float val = x[idx] + bias[0];
        val = val / div_value;
        float sig = 1.0f / (1.0f + expf(-val));
        out[idx] = val * sig;
    }
}

torch::Tensor fused_bias_div_swish_cuda(
    torch::Tensor x,
    torch::Tensor bias,
    float div_value)
{
    auto total_elements = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_bias_div_swish_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        div_value,
        out.data_ptr<float>(),
        total_elements);

    return out;
}
"""

fused_bias_div_swish_cpp_source = """
torch::Tensor fused_bias_div_swish_cuda(
    torch::Tensor x,
    torch::Tensor bias,
    float div_value);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_bias_div_swish",
    cpp_sources=fused_bias_div_swish_cpp_source,
    cuda_sources=fused_bias_div_swish_source,
    functions=["fused_bias_div_swish_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model with a fused CUDA kernel for bias addition, division, and Swish activation.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.divide_value = divide_value
        self.fused_op = fused_op

    def forward(self, x):
        x = self.matmul(x)
        x = self.bn(x)
        x = self.fused_op.fused_bias_div_swish_cuda(x, self.bias, self.divide_value)
        return x