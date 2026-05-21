import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation:
# 1. ConvTranspose3d (simplified assumption: we will replace the whole forward pass logic 
#    or specific heavy parts. However, replacing ConvTranspose3d entirely with a custom 
#    kernel is complex due to variable shapes. A better strategy for "speedup" in this 
#    specific small model is to fuse the element-wise operations after the conv, 
#    as the conv itself is already highly optimized in cuDNN/cuBLAS.
#    
#    However, the prompt asks to replace operators to get speedups. The element-wise 
#    sequence: x + bias, x + original_x, x * original_x, x + original_x involves multiple 
#    memory reads/writes. We can fuse these into a single kernel that takes the output 
#    of ConvTranspose3d and the bias/original_x to perform all arithmetic in one pass.
#
#    Let's define a kernel that performs:
#    out = (x + bias) + original_x
#    out = out * original_x
#    out = out + original_x
#    
#    Wait, looking at the code:
#    x = self.conv_transpose(x) -> let this be 'conv_out'
#    original_x = conv_out.clone().detach()
#    x = conv_out + bias
#    x = x + original_x  => (conv_out + bias) + conv_out = 2*conv_out + bias
#    x = x * original_x  => (2*conv_out + bias) * conv_out
#    x = x + original_x  => ((2*conv_out + bias) * conv_out) + conv_out
    
#    We can fuse the element-wise part. The ConvTranspose3d remains standard PyTorch 
#    for correctness and stability, but we replace the subsequent chain of operations 
#    with a custom fused kernel to reduce memory bandwidth overhead.

fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_elementwise_kernel(
    const float* conv_out, 
    const float* bias, 
    float* out, 
    int num_elements,
    int channels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        // Calculate channel index for broadcasting bias
        int c_idx = idx / (num_elements / channels);
        
        float x_val = conv_out[idx];
        float b_val = bias[c_idx];
        
        // Step 1: x + bias
        float step1 = x_val + b_val;
        
        // Step 2: step1 + original_x (which is x_val)
        float step2 = step1 + x_val; // Equivalent to 2*x_val + b_val
        
        // Step 3: step2 * original_x (which is x_val)
        float step3 = step2 * x_val;
        
        // Step 4: step3 + original_x (which is x_val)
        out[idx] = step3 + x_val;
    }
}

torch::Tensor fused_elementwise_cuda(torch::Tensor conv_out, torch::Tensor bias) {
    auto num_elements = conv_out.numel();
    auto channels = conv_out.size(1); // Assuming NCHW format for 3D data
    
    auto out = torch::empty_like(conv_out);

    const int block_size = 256;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    fused_elementwise_kernel<<<num_blocks, block_size>>>(
        conv_out.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        out.data_ptr<float>(), 
        num_elements,
        channels
    );

    return out;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_elementwise_cuda(torch::Tensor conv_out, torch::Tensor bias);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_elementwise",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_elementwise_cuda"],
    verbose=False,
    extra_cflags=["-O2"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution, 
    followed by a fused element-wise operation replacing the sum, residual add, 
    multiplication, and final residual add.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # Perform the convolution
        conv_out = self.conv_transpose(x)
        
        # Use the custom fused CUDA operator for the subsequent element-wise operations
        # This replaces:
        # original_x = conv_out.clone().detach()
        # x = conv_out + self.bias
        # x = x + original_x
        # x = x * original_x
        # x = x + original_x
        
        out = fused_ops.fused_elementwise_cuda(conv_out, self.bias)
        
        return out

# Re-defining get_inputs and get_init_inputs for completeness as per the prompt structure
def get_inputs():
    return [torch.rand(16, 32, 16, 32, 32)]

def get_init_inputs():
    return [32, 64, 3, 2, 1, 1, (64, 1, 1, 1)]