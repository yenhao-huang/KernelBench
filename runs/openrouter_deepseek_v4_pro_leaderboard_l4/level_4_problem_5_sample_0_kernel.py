import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from torch.utils.cpp_extension import load_inline

# CUDA source code for custom GELU and LayerNorm kernels
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Approximate GELU activation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
__global__ void gelu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
        output[idx] = x * cdf;
    }
}

torch::Tensor gelu_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    gelu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    return output;
}

// Fused LayerNorm kernel: computes mean and variance, normalizes, applies affine transform
__global__ void layernorm_kernel(const float* input, float* output, const float* weight, const float* bias,
                                 int hidden_size, float eps) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int row = blockIdx.x;
    int idx = row * hidden_size + tid;

    // Load input into shared memory, pad with zeros for threads beyond hidden_size
    float val = (tid < hidden_size) ? input[idx] : 0.0f;
    shared[tid] = val;
    __syncthreads();

    // Parallel reduction for mean
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    float mean = shared[0] / hidden_size;

    // Compute (x - mean)^2 and store in shared
    if (tid < hidden_size) {
        float diff = input[idx] - mean;
        shared[tid] = diff * diff;
    } else {
        shared[tid] = 0.0f;
    }
    __syncthreads();

    // Parallel reduction for variance
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    float variance = shared[0] / hidden_size;

    // Normalize and apply affine
    if (tid < hidden_size) {
        float inv_std = rsqrtf(variance + eps);
        float normalized = (input[idx] - mean) * inv_std;
        output[idx] = normalized * weight[tid] + bias[tid];
    }
}

torch::Tensor layernorm_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int hidden_size, float eps) {
    // input is assumed to be 2D: (N, hidden_size)
    int N = input.size(0);
    auto output = torch::zeros_like(input);
    const int block_size = 1024;  // must be >= hidden_size, power of 2 for reduction
    dim3 grid(N);
    dim3 block(block_size);
    size_t shared_mem = block_size * sizeof(float);
    layernorm_kernel<<<grid, block, shared_mem>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        weight.data_ptr<float>(), bias.data_ptr<float>(),
        hidden_size, eps);
    return output;
}
"""

cpp_source = """
torch::Tensor gelu_cuda(torch::Tensor input);
torch::Tensor layernorm_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int hidden_size, float eps);
"""

# Compile the inline CUDA code
custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["gelu_cuda", "layernorm_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Custom autograd Functions (inference only, no backward)
class GELUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return custom_ops.gelu_cuda(input)

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Training not supported")

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, hidden_size, eps):
        # Flatten all leading dimensions into one
        original_shape = input.shape
        input_2d = input.reshape(-1, hidden_size)
        output_2d = custom_ops.layernorm_cuda(input_2d, weight, bias, hidden_size, eps)
        return output_2d.reshape(original_shape)

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Training not supported")

# Custom modules that replace nn.GELU and nn.LayerNorm
class CustomGELU(nn.Module):
    def forward(self, x):
        return GELUFunction.apply(x)

class CustomLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # Assume last dim is the normalized shape
        hidden_size = self.normalized_shape[0]
        weight = self.weight if self.elementwise_affine else torch.ones(hidden_size, device=x.device, dtype=x.dtype)
        bias = self.bias if self.elementwise_affine else torch.zeros(hidden_size, device=x.device, dtype=x.dtype)
        return LayerNormFunction.apply(x, weight, bias, hidden_size, self.eps)

# Recursively replace modules in the given model
def replace_modules(module):
    for name, child in module.named_children():
        if isinstance(child, nn.GELU):
            setattr(module, name, CustomGELU())
        elif isinstance(child, nn.LayerNorm):
            custom_ln = CustomLayerNorm(child.normalized_shape, child.eps, child.elementwise_affine)
            if child.elementwise_affine:
                custom_ln.weight = child.weight
                custom_ln.bias = child.bias
            setattr(module, name, custom_ln)
        else:
            replace_modules(child)

class ModelNew(nn.Module):
    def __init__(self, model_name, config):
        super().__init__()
        self.model_name = model_name
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, config=self.config)
        # Replace GELU and LayerNorm with custom CUDA kernels
        replace_modules(self.model)

    def forward(self, x):
        return self.model(x).logits