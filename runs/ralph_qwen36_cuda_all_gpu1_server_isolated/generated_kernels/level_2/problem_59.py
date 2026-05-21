import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Matmul + Swish + Scale
# This fuses: Linear (Matmul + Bias), Swish (x * sigmoid(x)), and Scaling.
# We assume bias is used in the linear layer for generality, but can be zeroed if not needed.
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Helper to compute sigmoid efficiently
__device__ inline float fast_sigmoid(float x) {
    // Approximation or standard exp-based. For high precision, use exp.
    // 1 / (1 + exp(-x))
    return 1.0f / (1.0f + expf(-x));
}

__global__ void fused_matmul_swish_scale_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ output, 
    int batch_size, 
    int in_features, 
    int out_features,
    float scaling_factor
) {
    // Each thread handles one element of the output matrix (batch_idx, out_idx)
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size && out_idx < out_features) {
        float sum = 0.0f;
        
        // Perform the dot product for this specific output neuron
        #pragma unroll
        for (int i = 0; i < in_features; ++i) {
            sum += input[batch_idx * in_features + i] * weight[out_idx * in_features + i];
        }

        // Add bias if provided (assuming bias is not null, otherwise pass nullptr or handle separately)
        if (bias != nullptr) {
            sum += bias[out_idx];
        }

        // Apply Swish: x * sigmoid(x)
        float sigmoid_val = fast_sigmoid(sum);
        float swished = sum * sigmoid_val;

        // Apply scaling factor
        output[batch_idx * out_features + out_idx] = swished * scaling_factor;
    }
}

torch::Tensor fused_matmul_swish_scale_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    float scaling_factor
) {
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);

    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(weight.dim() == 2, "Weight must be 2D");

    auto output = torch::zeros({batch_size, out_features}, input.options());

    const int block_x = 32;
    const int block_y = 8; // 32*8 = 256 threads per block
    
    dim3 block(block_x, block_y);
    dim3 grid((out_features + block_x - 1) / block_x, (batch_size + block_y - 1) / block_y);

    const float* bias_ptr = bias.numel() > 0 ? bias.data_ptr<float>() : nullptr;

    fused_matmul_swish_scale_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        scaling_factor
    );

    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_matmul_swish_scale_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float scaling_factor);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_matmul_swish_scale_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Optimized model using a custom fused CUDA operator for Matmul + Swish + Scale.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scaling_factor = scaling_factor
        
        # Initialize weights and biases manually to match nn.Linear behavior
        # nn.Linear default initialization: uniform(-sqrt(1/in_features), sqrt(1/in_features))
        bound = 1.0 / (in_features ** 0.5)
        self.weight = nn.Parameter(torch.empty(out_features, in_features).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # Call the custom fused kernel
        return fused_ops.fused_matmul_swish_scale_cuda(x, self.weight, self.bias, self.scaling_factor)


# Helper functions to match the interface expected by the prompt structure
def get_inputs():
    batch_size = 128
    in_features = 32768
    out_features = 32768
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    in_features = 32768
    out_features = 32768
    scaling_factor = 2.0
    return [in_features, out_features, scaling_factor]