import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Conv2d + ReLU + HardSwish fusion
# This kernel performs: out = x * clamp((x + 3) / 6, 0, 1) where x is the result of conv(x)
# However, since we are replacing the PyTorch operators, we need to implement the Convolution part as well.
# Note: Implementing a full optimized Conv2d from scratch in inline CUDA is extremely complex and error-prone 
# compared to using cuDNN via torch.nn.functional.conv2d. 
# The prompt asks to replace pytorch operators with custom CUDA operators for speedups.
# A common optimization pattern is fusing the activation functions after a linear/conv layer.
# Here, we will fuse ReLU and HardSwish into a single kernel that operates on the output of the convolution.
# We will keep the convolution as a standard PyTorch operation (which uses cuDNN) but fuse the subsequent activations.
# Alternatively, to strictly follow "replace pytorch operators", we could replace the entire forward pass logic 
# with a custom kernel if we assume the weights are passed in. But typically, "replacing operators" implies 
# replacing specific function calls like relu or hardswish with faster fused versions.

# Let's define a fused activation kernel: ReLU followed by HardSwish.
# Note: HardSwish(x) = x * clamp((x + 3) / 6, 0, 1). 
# If we apply ReLU first: y = max(0, x). Then HardSwish(y) = y * clamp((y + 3) / 6, 0, 1).
# Since y >= 0, (y+3)/6 is always >= 0.5, so the clamp lower bound 0 is redundant if y>=0.
# Also since y can be large, we need to check upper bound? No, HardSwish doesn't have an upper bound on input, 
# but the clamp limits the multiplier.
# Actually, standard HardSwish is defined as x * min(max((x+3)/6, 0), 1).
# If we fuse ReLU and HardSwish:
# Step 1: x_relu = max(0, x)
# Step 2: out = x_relu * clamp((x_relu + 3) / 6, 0, 1)

fused_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_relu_hardswish_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        
        // ReLU
        float relu_out = fmaxf(0.0f, x);
        
        // HardSwish: y * clamp((y + 3) / 6, 0, 1)
        // Since relu_out >= 0, (relu_out + 3) / 6 >= 0.5, so lower bound 0 is always satisfied.
        float val = (relu_out + 3.0f) / 6.0f;
        float clamp_val = fminf(1.0f, val); // Upper bound check
        
        output[idx] = relu_out * clamp_val;
    }
}

torch::Tensor fused_relu_hardswish_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_relu_hardswish_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

fused_activation_cpp_source = (
    "torch::Tensor fused_relu_hardswish_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for fused activation
fused_activation = load_inline(
    name="fused_relu_hardswish",
    cpp_sources=fused_activation_cpp_source,
    cuda_sources=fused_activation_source,
    functions=["fused_relu_hardswish_cuda"],
    verbose=False,
    extra_cflags=["-O2"],
    extra_ldflags=["-lcudart"],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution, then applies a fused ReLU + HardSwish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # We don't need to store the fused operator as a module attribute if we just call it directly,
        # but for consistency with the example structure, we can attach it.
        self.fused_act = fused_activation

    def forward(self, x):
        x = self.conv(x)
        # Apply fused ReLU + HardSwish
        x = self.fused_act.fused_relu_hardswish_cuda(x)
        return x


def get_inputs():
    # randomly generate input tensors based on the model architecture
    batch_size = 128
    in_channels = 8
    height, width = 128, 128
    return [torch.rand(batch_size, in_channels, height, width).cuda()]


def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    in_channels = 8
    out_channels = 64
    kernel_size = 3
    return [in_channels, out_channels, kernel_size]