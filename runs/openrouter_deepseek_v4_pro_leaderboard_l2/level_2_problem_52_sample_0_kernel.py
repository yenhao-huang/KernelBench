import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for Mish activation
mish_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void mish_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        // Numerically stable softplus: max(x,0) + log1p(exp(-|x|))
        float sp = fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x)));
        output[idx] = x * tanhf(sp);
    }
}

torch::Tensor mish_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    mish_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

mish_cpp_source = "torch::Tensor mish_cuda(torch::Tensor input);"

# Compile the inline CUDA code for Mish
mish_op = load_inline(
    name="mish_op",
    cpp_sources=mish_cpp_source,
    cuda_sources=mish_cuda_source,
    functions=["mish_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.mish = mish_op

    def forward(self, x):
        x = self.conv(x)
        x = self.mish.mish_cuda(x)
        x = self.bn(x)
        return x