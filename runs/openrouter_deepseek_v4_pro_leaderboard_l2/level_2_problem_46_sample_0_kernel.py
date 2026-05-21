import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused subtract, tanh, subtract
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_sub_tanh_sub_kernel(const float* input, float* output, float sub1, float sub2, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float val = input[idx] - sub1;
        val = tanhf(val);
        val = val - sub2;
        output[idx] = val;
    }
}

torch::Tensor fused_sub_tanh_sub_cuda(torch::Tensor input, float sub1, float sub2) {
    auto total_elements = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_sub_tanh_sub_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), sub1, sub2, total_elements
    );

    return output;
}
"""

fused_ops_cpp_source = "torch::Tensor fused_sub_tanh_sub_cuda(torch::Tensor input, float sub1, float sub2);"

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_sub_tanh_sub",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_sub_tanh_sub_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.avgpool = nn.AvgPool2d(kernel_size_pool)
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_ops.fused_sub_tanh_sub_cuda(x, self.subtract1_value, self.subtract2_value)
        x = self.avgpool(x)
        return x