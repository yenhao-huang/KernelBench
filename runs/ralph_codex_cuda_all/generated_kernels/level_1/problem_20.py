import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

leaky_relu_cpp_source = """
torch::Tensor leaky_relu_cuda(torch::Tensor x, double negative_slope);
"""

leaky_relu_cuda_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <stdint.h>

__global__ void leaky_relu_vec4_kernel(const float* __restrict__ x, float* __restrict__ out, int64_t n4, float slope) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        float4 v = reinterpret_cast<const float4*>(x)[idx];
        v.x = v.x >= 0.0f ? v.x : v.x * slope;
        v.y = v.y >= 0.0f ? v.y : v.y * slope;
        v.z = v.z >= 0.0f ? v.z : v.z * slope;
        v.w = v.w >= 0.0f ? v.w : v.w * slope;
        reinterpret_cast<float4*>(out)[idx] = v;
    }
}

__global__ void leaky_relu_tail_kernel(const float* __restrict__ x, float* __restrict__ out, int64_t start, int64_t n, float slope) {
    int64_t idx = start + (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = x[idx];
        out[idx] = v >= 0.0f ? v : v * slope;
    }
}

torch::Tensor leaky_relu_cuda(torch::Tensor x, double negative_slope) {
    auto out = torch::empty_like(x);
    int64_t n = x.numel();
    if (n == 0) {
        return out;
    }

    const int threads = 256;
    float slope = static_cast<float>(negative_slope);
    uintptr_t xp = reinterpret_cast<uintptr_t>(x.data_ptr<float>());
    uintptr_t op = reinterpret_cast<uintptr_t>(out.data_ptr<float>());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (((xp | op) & 15) == 0) {
        int64_t n4 = n >> 2;
        if (n4 > 0) {
            int blocks4 = static_cast<int>((n4 + threads - 1) / threads);
            leaky_relu_vec4_kernel<<<blocks4, threads, 0, stream>>>(x.data_ptr<float>(), out.data_ptr<float>(), n4, slope);
        }
        int64_t start = n4 << 2;
        if (start < n) {
            int blocks = static_cast<int>(((n - start) + threads - 1) / threads);
            leaky_relu_tail_kernel<<<blocks, threads, 0, stream>>>(x.data_ptr<float>(), out.data_ptr<float>(), start, n, slope);
        }
    } else {
        int blocks = static_cast<int>((n + threads - 1) / threads);
        leaky_relu_tail_kernel<<<blocks, threads, 0, stream>>>(x.data_ptr<float>(), out.data_ptr<float>(), 0, n, slope);
    }

    return out;
}
"""

_leaky_relu_ext = load_inline(
    name="custom_leaky_relu_kernelbench",
    cpp_sources=leaky_relu_cpp_source,
    cuda_sources=leaky_relu_cuda_source,
    functions=["leaky_relu_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope
        self.leaky_relu = _leaky_relu_ext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.leaky_relu.leaky_relu_cuda(x, self.negative_slope)