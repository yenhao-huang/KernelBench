import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

selu_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <stdint.h>

static constexpr float SELU_SCALE = 1.0507009873554804934193349852946f;
static constexpr float SELU_ALPHA = 1.6732632423543772848170429916717f;
static constexpr float SELU_SA = SELU_SCALE * SELU_ALPHA;

__device__ __forceinline__ float selu_f32(float v) {
    return v > 0.0f ? SELU_SCALE * v : SELU_SA * (__expf(v) - 1.0f);
}

__global__ void selu_scalar_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (; idx < n; idx += stride) {
        y[idx] = selu_f32(x[idx]);
    }
}

__global__ void selu_vec4_kernel(const float4* __restrict__ x, float4* __restrict__ y, int64_t n4) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (; idx < n4; idx += stride) {
        float4 v = x[idx];
        v.x = selu_f32(v.x);
        v.y = selu_f32(v.y);
        v.z = selu_f32(v.z);
        v.w = selu_f32(v.w);
        y[idx] = v;
    }
}

torch::Tensor selu_cuda(torch::Tensor x) {
    auto y = torch::empty_like(x);
    int64_t n = x.numel();

    const int threads = 256;
    const int max_blocks = 65535;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    uintptr_t xp = reinterpret_cast<uintptr_t>(x.data_ptr<float>());
    uintptr_t yp = reinterpret_cast<uintptr_t>(y.data_ptr<float>());

    if ((n % 4 == 0) && ((xp & 15) == 0) && ((yp & 15) == 0)) {
        int64_t n4 = n / 4;
        int blocks = (int)((n4 + threads - 1) / threads);
        if (blocks > max_blocks) blocks = max_blocks;
        selu_vec4_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()),
            reinterpret_cast<float4*>(y.data_ptr<float>()),
            n4
        );
    } else {
        int blocks = (int)((n + threads - 1) / threads);
        if (blocks > max_blocks) blocks = max_blocks;
        selu_scalar_kernel<<<blocks, threads, 0, stream>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    }

    return y;
}
"""

selu_cpp = "torch::Tensor selu_cuda(torch::Tensor x);"

selu_ext = load_inline(
    name="selu_inline_cuda_kernelbench",
    cpp_sources=selu_cpp,
    cuda_sources=selu_source,
    functions=["selu_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.selu_ext = selu_ext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.selu_ext.selu_cuda(x)