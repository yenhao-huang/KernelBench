import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

layernorm_cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void layernorm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    long long rows,
    long long n,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ double smem[];
    double* ssum = smem;
    double* ssq = smem + blockDim.x;

    const long long base = (long long)row * n;

    double sum = 0.0;
    double sq = 0.0;

    for (long long i = tid; i < n; i += blockDim.x) {
        float v = x[base + i];
        sum += (double)v;
        sq += (double)v * (double)v;
    }

    ssum[tid] = sum;
    ssq[tid] = sq;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            ssum[tid] += ssum[tid + stride];
            ssq[tid] += ssq[tid + stride];
        }
        __syncthreads();
    }

    double mean = ssum[0] / (double)n;
    double var = ssq[0] / (double)n - mean * mean;
    double inv_std = rsqrt(var + (double)eps);

    for (long long i = tid; i < n; i += blockDim.x) {
        float y = (float)(((double)x[base + i] - mean) * inv_std);
        out[base + i] = y * gamma[i] + beta[i];
    }
}

torch::Tensor layernorm_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, double eps) {
    auto out = torch::empty_like(x);
    long long n = gamma.numel();
    long long rows = x.numel() / n;

    const int threads = 256;
    const dim3 blocks(rows);
    size_t shmem = threads * 2 * sizeof(double);

    layernorm_kernel<<<blocks, threads, shmem>>>(
        x.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        out.data_ptr<float>(),
        rows,
        n,
        (float)eps
    );

    return out;
}
"""

layernorm_cpp_source = r"""
torch::Tensor layernorm_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, double eps);
"""

_layernorm_ext = load_inline(
    name="custom_layernorm_fp32_ext",
    cpp_sources=layernorm_cpp_source,
    cuda_sources=layernorm_cuda_source,
    functions=["layernorm_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super().__init__()
        self.normalized_shape = tuple(normalized_shape)
        self.eps = 1e-5
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _layernorm_ext.layernorm_cuda(x, self.weight, self.bias, self.eps)