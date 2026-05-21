import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Conv3d + Mish + Tanh fusion
# This kernel performs: y = tanh(mish(conv3d(x)))
# Note: mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
# To avoid overflow in exp(x), we use the stable formulation:
# softplus(x) = max(x, 0) + ln(1 + exp(-|x|))
# mish(x) = x * tanh(max(x, 0) + ln(1 + exp(-|x|)))

conv_mish_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Helper for stable softplus: max(x, 0) + log(1 + exp(-abs(x)))
__device__ inline float stable_softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return 0.0f;
    float abs_x = fabsf(x);
    return fmaxf(x, 0.0f) + log1pf(expf(-abs_x));
}

// Mish activation: x * tanh(softplus(x))
__device__ inline float mish(float x) {
    return x * tanhf(stable_softplus(x));
}

// Kernel for Conv3d followed by Mish and Tanh
// We assume the input is in NCHW format (Batch, Channels, Depth, Height, Width)
// The convolution is performed using im2col logic or direct computation.
// For simplicity and performance on small kernels (3x3x3), we use a direct tiled approach 
// or rely on cuDNN via torch native if possible, but here we write a custom kernel.
// However, writing a full optimized Conv3d from scratch in inline CUDA is extremely verbose.
// A more practical "custom operator" optimization for this specific chain is to fuse the 
// activation functions (Mish + Tanh) after the convolution, or fuse them if we assume 
// the conv output is available.
// Since writing a full high-performance Conv3d kernel from scratch in inline code is 
// error-prone and lengthy, we will focus on fusing Mish and Tanh into a single kernel 
// that takes the Conv3d output as input. This is a valid optimization step.
// Alternatively, we can use torch.nn.functional.conv3d for the conv part (which uses cuDNN) 
// and then fuse the activations.

__global__ void mish_tanh_kernel(const float* input, float* output, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float val = input[idx];
        // Compute Mish: x * tanh(softplus(x))
        float sp = stable_softplus(val);
        float m_val = val * tanhf(sp);
        // Compute Tanh of the result
        output[idx] = tanhf(m_val);
    }
}

torch::Tensor conv_mish_tanh_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, 
                                  int stride_d, int stride_h, int stride_w, 
                                  int pad_d, int pad_h, int pad_w) {
    // Perform Conv3d using PyTorch's native implementation (which uses cuDNN for speed)
    // Then apply the fused Mish+Tanh activation.
    
    auto conv_out = torch::conv3d(x, weight, bias, 
                                  {stride_d, stride_h, stride_w}, 
                                  {pad_d, pad_h, pad_w});
    
    auto size = conv_out.numel();
    auto out = torch::empty_like(conv_out);
    
    if (size == 0) return out;
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    mish_tanh_kernel<<<num_blocks, block_size>>>(conv_out.data_ptr<float>(), out.data_ptr<float>(), size);
    
    return out;
}
"""

conv_mish_tanh_cpp_source = (
    "torch::Tensor conv_mish_tanh_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, "
    "int stride_d, int stride_h, int stride_w, int pad_d, int pad_h, int pad_w);"
)

# Compile the inline CUDA code
conv_mish_tanh = load_inline(
    name="conv_mish_tanh",
    cpp_sources=conv_mish_tanh_cpp_source,
    cuda_sources=conv_mish_tanh_source,
    functions=["conv_mish_tanh_cuda"],
    verbose=False,
    extra_cflags=["-O2"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D convolution, applies Mish activation, 
    and then applies Tanh activation using a custom fused CUDA operator.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ModelNew, self).__init__()
        # We need to store the parameters to pass them to the custom kernel
        # However, nn.Conv3d creates buffers. We will extract them or re-create them.
        # To make it a proper module, we can keep the conv layers but override forward.
        # Or we can just use the raw tensors passed in get_init_inputs.
        # The problem statement says "replace pytorch operators". 
        # We will initialize the weights here to match the original model's structure.
        
        self.conv_weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.conv_bias = nn.Parameter(torch.zeros(out_channels))
        
        self.stride_d = stride if isinstance(stride, int) else stride[0]
        self.stride_h = stride if isinstance(stride, int) else stride[1] if len(stride) > 1 else stride[0]
        self.stride_w = stride if isinstance(stride, int) else stride[-1] if isinstance(stride, tuple) else stride
        
        # Handle padding
        if isinstance(padding, int):
            self.pad_d = padding
            self.pad_h = padding
            self.pad_w = padding
        elif isinstance(padding, tuple):
            self.pad_d = padding[0]
            self.pad_h = padding[1]
            self.pad_w = padding[2] if len(padding) > 2 else padding[1]
        else:
            self.pad_d = 0
            self.pad_h = 0
            self.pad_w = 0

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D', H', W').
        """
        return conv_mish_tanh.conv_mish_tanh_cuda(
            x, 
            self.conv_weight, 
            self.conv_bias,
            self.stride_d, 
            self.stride_h, 
            self.stride_w,
            self.pad_d, 
            self.pad_h, 
            self.pad_w
        )

def get_inputs():
    return [torch.rand(16, 32, 32, 64, 64)]

def get_init_inputs():
    return [32, 64, 3]