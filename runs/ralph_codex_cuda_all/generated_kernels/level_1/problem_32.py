import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

hardtanh_cpp_source = """
torch::Tensor hardtanh_cuda(torch::Tensor x);
"""

hardtanh_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ float clamp_hardtanh(float v) {
    return fminf(1.0f, fmaxf(-1.0f, v));
}

__global__ void hardtanh_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = clamp_hardtanh(x[idx]);
    }
}

__global__ void hardtanh_vec4_kernel(const float4* __restrict__ x, float4* __restrict__ y, int64_t n4) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        float4 v = x[idx];
        v.x = clamp_hardtanh(v.x);
        v.y = clamp_hardtanh(v.y);
        v.z = clamp_hardtanh(v.z);
        v.w = clamp_hardtanh(v.w);
        y[idx] = v;
    }
}

torch::Tensor hardtanh_cuda(torch::Tensor x) {
    auto y = torch::empty_like(x);
    int64_t n = x.numel();
    if (n == 0) {
        return y;
    }

    const int threads = 256;
    uintptr_t xp = reinterpret_cast<uintptr_t>(x.data_ptr<float>());
    uintptr_t yp = reinterpret_cast<uintptr_t>(y.data_ptr<float>());

    if (((xp | yp) & 15) == 0 && (n & 3) == 0) {
        int64_t n4 = n >> 2;
        int blocks = (int)((n4 + threads - 1) / threads);
        hardtanh_vec4_kernel<<<blocks, threads>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()),
            reinterpret_cast<float4*>(y.data_ptr<float>()),
            n4
        );
    } else {
        int blocks = (int)((n + threads - 1) / threads);
        hardtanh_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    }

    return y;
}
"""

hardtanh_ext = load_inline(
    name="hardtanh_cuda_ext",
    cpp_sources=hardtanh_cpp_source,
    cuda_sources=hardtanh_cuda_source,
    functions=["hardtanh_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.hardtanh_ext = hardtanh_ext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hardtanh_ext.hardtanh_cuda(x)