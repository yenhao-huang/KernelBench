import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

masked_cumsum_cpp_source = """
torch::Tensor masked_cumsum_cuda(torch::Tensor x, torch::Tensor mask, int64_t dim);
"""

masked_cumsum_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

__global__ void masked_cumsum_dim1_kernel(
    const float* __restrict__ x,
    const bool* __restrict__ mask,
    float* __restrict__ out,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    extern __shared__ float shared[];
    int chunk = (cols + blockDim.x - 1) / blockDim.x;
    int base = row * cols;
    int start = tid * chunk;
    int end = start + chunk;
    if (end > cols) end = cols;

    float local_sum = 0.0f;
    for (int c = start; c < end; ++c) {
        local_sum += mask[base + c] ? x[base + c] : 0.0f;
    }

    shared[tid] = local_sum;
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        float v = 0.0f;
        if (tid >= offset) v = shared[tid - offset];
        __syncthreads();
        shared[tid] += v;
        __syncthreads();
    }

    float prefix = (tid == 0) ? 0.0f : shared[tid - 1];
    float acc = prefix;
    for (int c = start; c < end; ++c) {
        acc += mask[base + c] ? x[base + c] : 0.0f;
        out[base + c] = acc;
    }
}

torch::Tensor masked_cumsum_cuda(torch::Tensor x, torch::Tensor mask, int64_t dim) {
    auto out = torch::empty_like(x);

    int64_t nd = x.dim();
    if (dim < 0) dim += nd;

    int64_t cols64 = x.size(dim);
    int64_t rows64 = x.numel() / cols64;

    const int threads = 1024;
    dim3 block(threads);
    dim3 grid((unsigned int)rows64);
    size_t smem = threads * sizeof(float);

    masked_cumsum_dim1_kernel<<<grid, block, smem>>>(
        x.data_ptr<float>(),
        mask.data_ptr<bool>(),
        out.data_ptr<float>(),
        (int)rows64,
        (int)cols64
    );

    return out;
}
"""

masked_cumsum_ext = load_inline(
    name="masked_cumsum_ext",
    cpp_sources=masked_cumsum_cpp_source,
    cuda_sources=masked_cumsum_cuda_source,
    functions=["masked_cumsum_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.masked_cumsum_ext = masked_cumsum_ext

    def forward(self, x, mask):
        return self.masked_cumsum_ext.masked_cumsum_cuda(x, mask, self.dim)