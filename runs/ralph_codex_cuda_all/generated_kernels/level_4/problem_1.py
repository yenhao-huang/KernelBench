import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from torch.utils.cpp_extension import load_inline

gelu_cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math_constants.h>

__global__ void new_gelu_kernel(const float* __restrict__ x, float* __restrict__ y, long n) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = x[idx];
        float c = 0.7978845608028654f;
        float t = c * (v + 0.044715f * v * v * v);
        y[idx] = 0.5f * v * (1.0f + tanhf(t));
    }
}

torch::Tensor new_gelu_cuda(torch::Tensor x) {
    auto y = torch::empty_like(x);
    long n = x.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    new_gelu_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}
"""

gelu_cpp_source = r"""
torch::Tensor new_gelu_cuda(torch::Tensor x);
"""

gelu_ext = load_inline(
    name="gptneo_new_gelu_cuda_ext",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_cuda_source,
    functions=["new_gelu_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class CudaNewGELU(nn.Module):
    def forward(self, x):
        return gelu_ext.new_gelu_cuda(x)


class ModelNew(torch.nn.Module):
    def __init__(self, model_name, config):
        super().__init__()
        self.model_name = model_name
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, config=self.config)
        self.cuda_gelu = CudaNewGELU()

        for module in self.model.modules():
            if hasattr(module, "act"):
                module.act = self.cuda_gelu

    def forward(self, x):
        return self.model(x).logits