import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the GELU activation (approximation)
gelu_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void gelu_kernel(const float* __restrict__ x, float* __restrict__ out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        float cube = val * val * val;
        float inner = sqrtf(2.0f / M_PI) * (val + 0.044715f * cube);
        float tanh_inner = tanhf(inner);
        out[idx] = 0.5f * val * (1.0f + tanh_inner);
    }
}

torch::Tensor gelu_cuda(torch::Tensor x) {
    // Ensure input is contiguous and of float type
    x = x.contiguous().to(torch::kFloat32);
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

gelu_cpp_source = "torch::Tensor gelu_cuda(torch::Tensor x);"

# Compile the inline CUDA code for the GELU operator
gelu_op = load_inline(
    name="gelu_op",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_cuda_source,
    functions=["gelu_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized GELU activation using a fused CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gelu = gelu_op

    def forward(self, x):
        return self.gelu.gelu_cuda(x)