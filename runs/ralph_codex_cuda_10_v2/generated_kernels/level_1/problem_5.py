import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

scalar_mul_cpp_source = """
torch::Tensor scalar_mul_cuda(torch::Tensor A, double s);
"""

scalar_mul_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

__global__ void scalar_mul_kernel(const float* __restrict__ A, float* __restrict__ C, float s, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] * s;
    }
}

__global__ void scalar_mul_vec4_kernel(const float4* __restrict__ A, float4* __restrict__ C, float s, int64_t n4) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        float4 v = A[idx];
        v.x *= s;
        v.y *= s;
        v.z *= s;
        v.w *= s;
        C[idx] = v;
    }
}

torch::Tensor scalar_mul_cuda(torch::Tensor A, double s) {
    auto C = torch::empty_like(A);
    int64_t n = A.numel();
    const int threads = 256;
    float sf = static_cast<float>(s);

    const float* Ap = A.data_ptr<float>();
    float* Cp = C.data_ptr<float>();

    bool aligned = ((reinterpret_cast<uintptr_t>(Ap) & 15) == 0) &&
                   ((reinterpret_cast<uintptr_t>(Cp) & 15) == 0) &&
                   ((n & 3) == 0);

    if (aligned) {
        int64_t n4 = n >> 2;
        int blocks = static_cast<int>((n4 + threads - 1) / threads);
        scalar_mul_vec4_kernel<<<blocks, threads>>>(
            reinterpret_cast<const float4*>(Ap),
            reinterpret_cast<float4*>(Cp),
            sf,
            n4
        );
    } else {
        int blocks = static_cast<int>((n + threads - 1) / threads);
        scalar_mul_kernel<<<blocks, threads>>>(Ap, Cp, sf, n);
    }

    return C;
}
"""

scalar_mul_ext = load_inline(
    name="scalar_mul_ext_fp32",
    cpp_sources=scalar_mul_cpp_source,
    cuda_sources=scalar_mul_cuda_source,
    functions=["scalar_mul_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.scalar_mul_ext = scalar_mul_ext

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        return self.scalar_mul_ext.scalar_mul_cuda(A, s)