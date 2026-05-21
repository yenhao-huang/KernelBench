import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernels for fused operations to optimize the pipeline.
# We will fuse: 
# 1. ConvTranspose3d + LeakyReLU (to avoid intermediate buffer allocation)
# 2. Element-wise multiplication with learnable parameter + LeakyReLU (fused activation)
# Note: MaxPool3d is already highly optimized in cuDNN/cuBLAS, so we leave it as is.

custom_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Helper for LeakyReLU
__device__ inline float leaky_relu(float x, float slope) {
    return x > 0 ? x : x * slope;
}

// Kernel 1: ConvTranspose3d + LeakyReLU
// This kernel assumes the convolution output is computed on-the-fly or we replace the standard conv.
// However, implementing a full custom ConvTranspose3d from scratch in inline CUDA is extremely complex 
// and error-prone compared to leveraging cuDNN via torch.nn.functional.conv_transpose3d which is already optimized.
// Instead, we focus on fusing the post-processing steps: LeakyReLU -> Mul -> LeakyReLU.
// And potentially fusing the first LeakyReLU with the ConvTranspose if we were writing raw conv, 
// but since we can't easily replace the heavy lifting of ConvTranspose3d efficiently in inline code without cuDNN bindings,
// we will optimize the element-wise chain: LeakyReLU -> Mul -> LeakyReLU.

// Actually, a better strategy for "speedups" in this specific small model is to fuse the 
// LeakyReLU -> Mul -> LeakyReLU sequence into a single kernel, as these are memory-bound operations on large tensors.
// Also, we can fuse the first LeakyReLU with the ConvTranspose if we assume we can call the underlying cuDNN op 
// and immediately apply activation, but standard PyTorch ops don't expose that easily in inline CUDA without writing a wrapper.
// Let's stick to fusing the element-wise operations which are significant overheads for small channels/depths relative to compute.

__global__ void fused_leaky_relu_mul_leaky_relu_kernel(
    const float* input, 
    const float* multiplier, 
    float* output, 
    int total_elements, 
    int out_channels,
    float slope) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        // Calculate spatial and channel indices
        // Tensor layout: [N, C, D, H, W]
        // We need to know the stride for channels. 
        // Assuming contiguous tensor.
        
        // To apply multiplier correctly: multiplier shape is [C, 1, 1, 1]
        // So we multiply input[i] by multiplier[channel_idx]
        
        int spatial_idx = idx / out_channels;
        int channel_idx = idx % out_channels;
        
        float val = input[idx];
        
        // First LeakyReLU
        val = leaky_relu(val, slope);
        
        // Multiply by learnable parameter (broadcasted)
        val *= multiplier[channel_idx];
        
        // Second LeakyReLU
        val = leaky_relu(val, slope);
        
        output[idx] = val;
    }
}

torch::Tensor fused_leaky_relu_mul_leaky_relu_cuda(
    torch::Tensor input, 
    torch::Tensor multiplier) 
{
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(multiplier.is_cuda(), "Multiplier must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 5, "Input must be 5D [N, C, D, H, W]");
    
    const int batch_size = input.size(0);
    const int out_channels = input.size(1);
    const int total_elements = input.numel();
    
    auto output = torch::empty_like(input);
    
    const float slope = 0.2f;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_leaky_relu_mul_leaky_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        multiplier.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        out_channels,
        slope
    );
    
    return output;
}

// Kernel 2: ConvTranspose3d + LeakyReLU (Fused)
// We will use the standard torch functional call for conv_transpose but wrap it in a custom op 
// that applies the activation immediately if we were to write a full wrapper. 
// However, since we are limited to inline CUDA, we cannot easily call cuDNN directly without headers.
// Therefore, the most impactful optimization here is the element-wise fusion above.
// We will also provide a simple kernel for MaxPool3d if needed, but PyTorch's is very fast.
// Let's just output the fused element-wise op as it's the most straightforward "custom CUDA" win 
// that avoids multiple memory writes/reads for the intermediate activations.

"""

custom_cpp_source = """
torch::Tensor fused_leaky_relu_mul_leaky_relu_cuda(torch::Tensor input, torch::Tensor multiplier);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=custom_cpp_source,
    cuda_sources=custom_cuda_source,
    functions=["fused_leaky_relu_mul_leaky_relu_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"]
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution, applies LeakyReLU, 
    multiplies by a learnable parameter, applies LeakyReLU again (fused with previous step), 
    and performs a max pooling operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super(ModelNew, self).__init__()
        # Keep the standard ConvTranspose3d as it is heavily optimized by cuDNN and 
        // writing a custom one from scratch in inline CUDA is not feasible for general cases.
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        # We remove the explicit LeakyReLU layers and handle them in the fused CUDA kernel
        self.max_pool = nn.MaxPool3d(kernel_size=2)

    def forward(self, x):
        # Step 1: ConvTranspose3d
        x = self.conv_transpose(x)
        
        # Step 2: Fused LeakyReLU -> Mul -> LeakyReLU
        # This replaces:
        # x = self.leaky_relu(x)
        # x = x * self.multiplier
        # x = self.leaky_relu(x)
        x = fused_ops.fused_leaky_relu_mul_leaky_relu_cuda(x, self.multiplier)
        
        # Step 3: Max Pooling (Standard PyTorch op is efficient enough)
        x = self.max_pool(x)
        
        return x

# Re-defining get_inputs and get_init_inputs to match the required interface for testing/usage context
def get_inputs():
    return [torch.rand(16, 16, 16, 32, 32)]

def get_init_inputs():
    return [16, 32, 3, 2, 1, 1, (32, 1, 1, 1)]