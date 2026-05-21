import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

inorm_cpp_source = """
torch::Tensor instance_norm2d_cuda(torch::Tensor x, double eps);
"""

inorm_cuda_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

__global__ void instance_norm2d_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int outer,
    int spatial,
    float eps
) {
    int oc = blockIdx.x;
    if (oc >= outer) return;

    const float* base = x + (long long)oc * spatial;
    float* out = y + (long long)oc * spatial;

    __shared__ float ssum[256];
    __shared__ float ssq[256];

    int tid = threadIdx.x;
    float sum = 0.0f;
    float sq = 0.0f;

    for (int i = tid; i < spatial; i += blockDim.x) {
        float v = base[i];
        sum += v;
        sq += v * v;
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

    float mean = ssum[0] / (float)spatial;
    float var = ssq[0] / (float)spatial - mean * mean;
    var = var > 0.0f ? var : 0.0f;
    float inv_std = rsqrtf(var + eps);

    for (int i = tid; i < spatial; i += blockDim.x) {
        out[i] = (base[i] - mean) * inv_std;
    }
}

torch::Tensor instance_norm2d_cuda(torch::Tensor x, double eps) {
    auto y = torch::empty_like(x);

    int n = (int)x.size(0);
    int c = (int)x.size(1);
    int h = (int)x.size(2);
    int w = (int)x.size(3);

    int outer = n * c;
    int spatial = h * w;

    const int threads = 256;
    dim3 blocks(outer);

    instance_norm2d_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        outer,
        spatial,
        (float)eps
    );

    return y;
}
"""

_instance_norm2d = load_inline(
    name="instance_norm2d_fp32_inline",
    cpp_sources=inorm_cpp_source,
    cuda_sources=inorm_cuda_source,
    functions=["instance_norm2d_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        self.eps = 1e-5
        self._op = _instance_norm2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._op.instance_norm2d_cuda(x, self.eps)