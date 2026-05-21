import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for fused min and subtract
fused_min_sub_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_min_sub_kernel(const float* input, float* output, float constant, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        val = fminf(val, constant);
        output[idx] = val - constant;
    }
}

torch::Tensor fused_min_sub_cuda(torch::Tensor input, float constant) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_min_sub_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), constant, size);

    return output;
}
"""

fused_min_sub_cpp_source = "torch::Tensor fused_min_sub_cuda(torch::Tensor input, float constant);"

# Compile the inline CUDA code
fused_min_sub = load_inline(
    name="fused_min_sub",
    cpp_sources=fused_min_sub_cpp_source,
    cuda_sources=fused_min_sub_source,
    functions=["fused_min_sub_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, constant):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.constant = nn.Parameter(torch.tensor(constant))
        self.fused_min_sub = fused_min_sub

    def forward(self, x):
        x = self.linear(x)
        x = self.fused_min_sub.fused_min_sub_cuda(x, self.constant.item())
        return x