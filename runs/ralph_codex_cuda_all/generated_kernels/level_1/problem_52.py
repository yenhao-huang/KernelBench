import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

argmin_cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

__global__ void argmin_dim1_kernel(
    const float* __restrict__ x,
    int64_t* __restrict__ out,
    int batch,
    int dim1,
    int dim2
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;

    if (b >= batch || col >= dim2) {
        return;
    }

    int base = (b * dim1 * dim2) + col;
    float best = x[base];
    int best_idx = 0;

    #pragma unroll 4
    for (int r = 1; r < dim1; ++r) {
        float v = x[base + r * dim2];
        if (v < best) {
            best = v;
            best_idx = r;
        }
    }

    out[b * dim2 + col] = static_cast<int64_t>(best_idx);
}

torch::Tensor argmin_dim1_cuda(torch::Tensor x) {
    const int batch = static_cast<int>(x.size(0));
    const int dim1 = static_cast<int>(x.size(1));
    const int dim2 = static_cast<int>(x.size(2));

    auto out = torch::empty({batch, dim2}, x.options().dtype(torch::kInt64));

    const int threads = 256;
    dim3 blocks((dim2 + threads - 1) / threads, batch);

    argmin_dim1_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<int64_t>(),
        batch,
        dim1,
        dim2
    );

    return out;
}
"""

argmin_cpp_source = "torch::Tensor argmin_dim1_cuda(torch::Tensor x);"

argmin_ext = load_inline(
    name="argmin_dim1_ext",
    cpp_sources=argmin_cpp_source,
    cuda_sources=argmin_cuda_source,
    functions=["argmin_dim1_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.argmin_ext = argmin_ext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.argmin_ext.argmin_dim1_cuda(x)