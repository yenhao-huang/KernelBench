import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

tanh_cpp_source = """
torch::Tensor tanh_cuda(torch::Tensor x);
"""

tanh_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math_constants.h>

__global__ void tanh_kernel_scalar(const float* __restrict__ x, float* __restrict__ out, long long n) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = idx; i < n; i += stride) {
        out[i] = tanhf(x[i]);
    }
}

__global__ void tanh_kernel_vec4(const float4* __restrict__ x, float4* __restrict__ out, long long n4) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = idx; i < n4; i += stride) {
        float4 v = x[i];
        v.x = tanhf(v.x);
        v.y = tanhf(v.y);
        v.z = tanhf(v.z);
        v.w = tanhf(v.w);
        out[i] = v;
    }
}

torch::Tensor tanh_cuda(torch::Tensor x) {
    auto out = torch::empty_like(x);
    long long n = x.numel();

    const int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;

    uintptr_t xp = reinterpret_cast<uintptr_t>(x.data_ptr<float>());
    uintptr_t op = reinterpret_cast<uintptr_t>(out.data_ptr<float>());

    if ((n % 4 == 0) && (xp % 16 == 0) && (op % 16 == 0)) {
        long long n4 = n / 4;
        int blocks4 = (int)((n4 + threads - 1) / threads);
        if (blocks4 > 65535) blocks4 = 65535;
        tanh_kernel_vec4<<<blocks4, threads>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()),
            reinterpret_cast<float4*>(out.data_ptr<float>()),
            n4
        );
    } else {
        tanh_kernel_scalar<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
    }

    return out;
}
"""

tanh_ext = load_inline(
    name="custom_tanh_fp32_ext",
    cpp_sources=tanh_cpp_source,
    cuda_sources=tanh_cuda_source,
    functions=["tanh_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tanh_ext = tanh_ext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tanh_ext.tanh_cuda(x)