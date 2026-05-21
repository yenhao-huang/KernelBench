import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Linear + Mish + Mish
# This fuses: matmul (x @ W.T + b), mish(x), mish(mish(x))
# to minimize memory bandwidth and launch overhead.

fused_linear_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Mish activation: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
__device__ __forceinline__ float mish(float x) {
    // For numerical stability, we can use the standard definition.
    // tanh(softplus(x)) where softplus(x) = ln(1 + exp(x))
    // However, for large positive x, exp(x) overflows. 
    // A stable implementation:
    // if x > threshold: x * 1.0 (approx) or use log1p(exp(-x)) trick?
    // Standard PyTorch mish handles this. Let's use a robust approximation.
    
    float softplus = log1p(expf(x)); // ln(1 + exp(x))
    return x * tanhf(softplus);
}

__global__ void fused_linear_mish_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ output, 
    int batch_size, 
    int in_features, 
    int out_features
) {
    // Each thread handles one element of the output matrix (batch, out_features)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size * out_features) return;
    
    int b = idx / out_features;
    int o = idx % out_features;
    
    float sum = 0.0f;
    
    // Perform dot product for this output neuron
    const float* input_row = input + b * in_features;
    const float* weight_col = weight + o * in_features; // Weight is stored as [out, in] in nn.Linear
    
    #pragma unroll
    for (int i = 0; i < in_features; ++i) {
        sum += input_row[i] * weight_col[i];
    }
    
    // Add bias
    if (bias != nullptr) {
        sum += bias[o];
    }
    
    // Apply Mish twice
    float m1 = mish(sum);
    float m2 = mish(m1);
    
    output[idx] = m2;
}

torch::Tensor fused_linear_mish_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, input.options());
    
    const int block_size = 256;
    const int total_elements = batch_size * out_features;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    // Check if bias is present
    float* bias_ptr = nullptr;
    if (bias.numel() > 0) {
        bias_ptr = bias.data_ptr<float>();
    }

    fused_linear_mish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\\n", cudaGetErrorString(err));
    }
    
    return output;
}
"""

fused_linear_mish_cpp_source = (
    "torch::Tensor fused_linear_mish_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_linear_mish",
    cpp_sources=fused_linear_mish_cpp_source,
    cuda_sources=fused_linear_mish_source,
    functions=["fused_linear_mish_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Optimized model using a custom fused CUDA kernel for Linear + Mish + Mish.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights and biases manually to match nn.Linear behavior
        # nn.Linear uses Kaiming uniform initialization by default for linear layers
        weight = torch.empty(out_features, in_features)
        bias = torch.empty(out_features)
        
        # Use the same initialization as nn.Linear
        stdv = 1. / math.sqrt(in_features) if hasattr(math, 'sqrt') else 1.0 / (in_features ** 0.5)
        # Actually, let's just use torch.nn.init.kaiming_uniform_ to be safe and consistent
        import torch.nn.init as init
        init.kaiming_uniform_(weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
        bound = 1. / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(bias, -bound, bound)
        
        self.register_buffer('weight', weight)
        self.register_buffer('bias', bias)

    def forward(self, x):
        # x: [batch_size, in_features]
        # weight: [out_features, in_features]
        # bias: [out_features]
        return fused_ops.fused_linear_mish_cuda(x, self.weight, self.bias)


import math

def get_inputs():
    return [torch.rand(1024, 8192)]

def get_init_inputs():
    return [8192, 8192]