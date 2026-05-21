import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from torch.utils.cpp_extension import load_inline

# ---------- Custom CUDA LayerNorm ----------
layernorm_cuda_src = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void layernorm_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    const int batch_size,
    const int hidden_size,
    const float eps
) {
    // Each block processes one row
    int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared_mem[];
    float* mean_shared = shared_mem;
    float* var_shared = shared_mem + blockDim.x;

    int tid = threadIdx.x;
    int stride = blockDim.x;
    int idx = row * hidden_size + tid;

    // Compute mean
    float sum = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        sum += input[row * hidden_size + i];
    }
    mean_shared[tid] = sum;
    __syncthreads();

    // Reduction for mean
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            mean_shared[tid] += mean_shared[tid + s];
        }
        __syncthreads();
    }
    float mean = mean_shared[0] / hidden_size;

    // Compute variance
    sum = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        float diff = input[row * hidden_size + i] - mean;
        sum += diff * diff;
    }
    var_shared[tid] = sum;
    __syncthreads();

    // Reduction for variance
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            var_shared[tid] += var_shared[tid + s];
        }
        __syncthreads();
    }
    float var = var_shared[0] / hidden_size;

    // Normalize
    float inv_std = rsqrtf(var + eps);
    for (int i = tid; i < hidden_size; i += stride) {
        float val = input[row * hidden_size + i];
        float norm = (val - mean) * inv_std;
        output[row * hidden_size + i] = norm * gamma[i] + beta[i];
    }
}

torch::Tensor custom_layernorm_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps
) {
    const int batch_size = input.size(0);
    const int hidden_size = input.size(1);
    auto output = torch::empty_like(input);

    const int block_dim = 256;
    const int shared_mem_size = block_dim * sizeof(float) * 2; // for mean and var reductions
    dim3 grid(batch_size);
    dim3 block(block_dim);

    layernorm_forward_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        hidden_size,
        eps
    );

    return output;
}
"""

layernorm_cpp_src = "torch::Tensor custom_layernorm_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, float eps);"

# Compile the custom CUDA LayerNorm
custom_layernorm = load_inline(
    name="custom_layernorm",
    cpp_sources=layernorm_cpp_src,
    cuda_sources=layernorm_cuda_src,
    functions=["custom_layernorm_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# ---------- Custom Module Wrapper ----------
class CustomLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps, weight, bias):
        super().__init__()
        self.weight = weight
        self.bias = bias
        self.eps = eps
        self.normalized_shape = normalized_shape

    def forward(self, x):
        # x: (batch_size, ..., hidden_size) for NLP typical use
        # Assuming input is 2D or 3D, we flatten to 2D for kernel, then reshape back
        orig_shape = x.shape
        if x.dim() > 2:
            x = x.reshape(-1, orig_shape[-1])
        out = custom_layernorm.custom_layernorm_cuda(x, self.weight, self.bias, self.eps)
        if x.dim() != orig_shape:
            out = out.reshape(orig_shape)
        return out

# ---------- Optimized Model ----------
class ModelNew(nn.Module):
    def __init__(self, model_name, config):
        super().__init__()
        self.model_name = model_name
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, config=self.config)
        self._replace_layernorms()

    def _replace_layernorms(self):
        # Helper to recursively replace nn.LayerNorm with CustomLayerNorm
        def replace_module(module, prefix=''):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                if isinstance(child, nn.LayerNorm):
                    # Create custom module with same parameters
                    custom_ln = CustomLayerNorm(
                        child.normalized_shape,
                        child.eps,
                        child.weight,
                        child.bias
                    )
                    setattr(module, name, custom_ln)
                else:
                    replace_module(child, full_name)
        replace_module(self.model)

    def forward(self, x):
        return self.model(x).logits