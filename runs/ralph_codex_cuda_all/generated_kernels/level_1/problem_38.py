import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

l1_norm_cpp_source = """
torch::Tensor l1_norm_cuda(torch::Tensor x);
"""

l1_norm_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>

__global__ void l1_norm_kernel(const float* __restrict__ x,
                               float* __restrict__ out,
                               int64_t rows,
                               int64_t cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_x = x + (int64_t)row * cols;
    float* row_out = out + (int64_t)row * cols;

    float sum = 0.0f;
    for (int64_t col = threadIdx.x; col < cols; col += blockDim.x) {
        sum += fabsf(row_x[col]);
    }

    __shared__ float smem[256];
    smem[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float inv_mean = ((float)cols) / smem[0];

    for (int64_t col = threadIdx.x; col < cols; col += blockDim.x) {
        row_out[col] = row_x[col] * inv_mean;
    }
}

torch::Tensor l1_norm_cuda(torch::Tensor x) {
    auto out = torch::empty_like(x);
    int64_t rows = x.size(0);
    int64_t cols = x.size(1);

    const int threads = 256;
    dim3 blocks((unsigned int)rows);

    l1_norm_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        rows,
        cols
    );

    return out;
}
"""

_l1_norm_module = load_inline(
    name="l1_norm_inline_cuda",
    cpp_sources=l1_norm_cpp_source,
    cuda_sources=l1_norm_cuda_source,
    functions=["l1_norm_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.l1_norm = _l1_norm_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l1_norm.l1_norm_cuda(x)