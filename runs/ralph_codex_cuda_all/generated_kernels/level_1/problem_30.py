import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

softsign_cpp_source = """
torch::Tensor softsign_cuda(torch::Tensor x);
"""

softsign_cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ float softsign_f32(float v) {
    return v / (1.0f + fabsf(v));
}

__global__ void softsign_scalar_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = softsign_f32(x[i]);
    }
}

__global__ void softsign_vec4_kernel(const float4* __restrict__ x, float4* __restrict__ y, int64_t n4) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 v = x[i];
        v.x = softsign_f32(v.x);
        v.y = softsign_f32(v.y);
        v.z = softsign_f32(v.z);
        v.w = softsign_f32(v.w);
        y[i] = v;
    }
}

torch::Tensor softsign_cuda(torch::Tensor x) {
    auto y = torch::empty_like(x);
    int64_t n = x.numel();

    const float* xp = x.data_ptr<float>();
    float* yp = y.data_ptr<float>();

    const int threads = 256;
    uintptr_t xa = reinterpret_cast<uintptr_t>(xp);
    uintptr_t ya = reinterpret_cast<uintptr_t>(yp);

    if ((n % 4 == 0) && ((xa & 15) == 0) && ((ya & 15) == 0)) {
        int64_t n4 = n / 4;
        int blocks = (int)((n4 + threads - 1) / threads);
        softsign_vec4_kernel<<<blocks, threads>>>(
            reinterpret_cast<const float4*>(xp),
            reinterpret_cast<float4*>(yp),
            n4
        );
    } else {
        int blocks = (int)((n + threads - 1) / threads);
        softsign_scalar_kernel<<<blocks, threads>>>(xp, yp, n);
    }

    return y;
}
"""

softsign_ext = load_inline(
    name="softsign_inline_cuda_ext",
    cpp_sources=softsign_cpp_source,
    cuda_sources=softsign_cuda_source,
    functions=["softsign_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softsign_ext = softsign_ext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softsign_ext.softsign_cuda(x)