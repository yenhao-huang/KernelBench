import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
from transformers import AutoModelForCausalLM, AutoConfig

# CUDA source for fast exact GELU
gelu_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_forward_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        output[idx] = 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
    }
}

torch::Tensor gelu_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    auto n = input.numel();
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;
    
    gelu_forward_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), n);
    
    return output;
}
"""

gelu_cpp_source = "torch::Tensor gelu_cuda(torch::Tensor input);"

# Compile the custom CUDA GELU
custom_gelu = load_inline(
    name="custom_gelu",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_cuda_source,
    functions=["gelu_cuda"],
    verbose=False,
)

# Autograd function wrapper (forward only, inference only)
class FastGeluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return custom_gelu.gelu_cuda(input)

    @staticmethod
    def backward(ctx, grad_output):
        # Not required for inference, but provided for completeness
        return None

class FastGelu(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return FastGeluFunction.apply(x)

# Recursively replace nn.GELU with FastGelu
def replace_gelu(module):
    for name, child in module.named_children():
        if isinstance(child, nn.GELU):
            setattr(module, name, FastGelu())
        else:
            replace_gelu(child)

class ModelNew(nn.Module):
    def __init__(self, model_name, config):
        super().__init__()
        self.model_name = model_name
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, config=self.config)
        replace_gelu(self.model)

    def forward(self, x):
        return self.model(x).logits