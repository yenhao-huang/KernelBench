import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Conv3d + ReLU + LeakyReLU + GELU + Sigmoid + Bias Add
# Note: Standard PyTorch Conv3d is already highly optimized. The main opportunity here is fusing the 
# activation chain and bias addition into a single pass to avoid global memory writes/reads between activations.
# We will implement a custom kernel that performs: y = GELU(Sigmoid(LeakyReLU(ReLU(Conv3d(x) + bias))))
# However, since Conv3d is complex to write from scratch efficiently in inline CUDA without libraries like CUTLASS,
# we will focus on fusing the post-convolution activations and bias addition. 
# To strictly follow "replace pytorch operators", we can replace the entire forward pass logic with a custom kernel 
# that calls the underlying conv implementation or implements a simplified version if needed. 
# But given the complexity of Conv3d, a more realistic optimization is to fuse the element-wise operations after conv.
# Let's implement a fused activation + bias kernel.

fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Helper for GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
__device__ inline float gelu(float x) {
    float cdf = 0.5f * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
    return x * cdf;
}

__global__ void fused_activations_bias_kernel(
    const float* input, 
    const float* bias, 
    float* output, 
    int batch_size, 
    int out_channels, 
    int spatial_size // depth * height * width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * spatial_size;

    if (idx < total_elements) {
        // Determine channel index for bias lookup
        int spatial_idx = idx % spatial_size;
        int channel_idx = (idx / spatial_size) % out_channels;
        
        float val = input[idx];
        
        // 1. ReLU
        if (val < 0) val = 0.0f;
        
        // 2. LeakyReLU (negative_slope=0.01)
        if (val < 0) val = 0.01f * val;
        
        // 3. GELU
        val = gelu(val);
        
        // 4. Sigmoid
        val = 1.0f / (1.0f + expf(-val));
        
        // 5. Add Bias
        // Bias shape is (out_channels, 1, 1, 1), so it broadcasts over batch and spatial dims
        float b = bias[channel_idx];
        output[idx] = val + b;
    }
}

torch::Tensor fused_ops_cuda(torch::Tensor x, torch::Tensor bias) {
    // x is expected to be (batch_size, out_channels, depth, height, width)
    // bias is expected to be (out_channels, 1, 1, 1)
    
    auto batch_size = x.size(0);
    auto out_channels = x.size(1);
    auto spatial_size = x.numel() / (batch_size * out_channels);
    
    auto output = torch::empty_like(x);
    
    const int block_size = 256;
    int total_elements = batch_size * out_channels * spatial_size;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_activations_bias_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        output.data_ptr<float>(), 
        batch_size, 
        out_channels, 
        spatial_size
    );

    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor x, torch::Tensor bias);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA operator for fused activations and bias addition.
    Note: The Conv3d operation is still performed by PyTorch's optimized C++/CUDA backend 
    because writing a high-performance generic 3D convolution from scratch in inline CUDA 
    is extremely complex and likely slower than cuDNN. The optimization focuses on fusing 
    the subsequent element-wise operations which are memory-bound.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 

    def forward(self, x):
        # Perform Conv3d using PyTorch's optimized backend
        x = self.conv(x)
        
        # Use custom fused CUDA operator for ReLU -> LeakyReLU -> GELU -> Sigmoid -> Bias Add
        x = fused_ops.fused_ops_cuda(x, self.bias)
        
        return x

# Keeping the input generation functions as requested by the prompt structure, 
# though the prompt asks to output the new model code. I will include them for completeness 
# if needed, but the prompt says "Just output the new model code". 
# However, the example included get_inputs and get_init_inputs. I will include them to match the format.

batch_size = 64
in_channels = 8
out_channels = 32
depth, height, width = 32, 64, 64
kernel_size = 3
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]