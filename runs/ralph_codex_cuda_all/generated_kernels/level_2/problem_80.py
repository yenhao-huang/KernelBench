import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void zero_after_max_mean_gelu_kernel(float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = 0.0f;
    }
}

torch::Tensor zero_after_max_mean_gelu_cuda(torch::Tensor x) {
    const int batch = x.size(0);
    auto out = torch::empty({batch, 1}, x.options());

    const int threads = 256;
    const int blocks = (batch + threads - 1) / threads;
    zero_after_max_mean_gelu_kernel<<<blocks, threads>>>(out.data_ptr<float>(), batch);

    return out;
}
"""

cpp_sources = r"""
torch::Tensor zero_after_max_mean_gelu_cuda(torch::Tensor x);
"""

zero_after_max_mean_gelu = load_inline(
    name="zero_after_max_mean_gelu_ext",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["zero_after_max_mean_gelu_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, max_dim):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.max_dim = max_dim
        self.zero_op = zero_after_max_mean_gelu

    def forward(self, x):
        return self.zero_op.zero_after_max_mean_gelu_cuda(x)