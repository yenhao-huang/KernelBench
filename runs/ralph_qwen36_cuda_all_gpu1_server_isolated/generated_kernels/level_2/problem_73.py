import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Conv2d + BatchNorm2d + Scale fusion.
# This kernel performs:
# 1. Convolution (im2col + gemm logic or direct convolution)
# 2. Batch Normalization (mean/variance normalization and affine transform)
# 3. Scaling by a constant factor
#
# To keep it robust and efficient without external libraries like cuDNN, 
# we will implement a simplified but optimized path. 
# Note: Writing a full high-performance Conv2d from scratch in inline CUDA is complex.
# However, for the purpose of this optimization task, we can leverage the fact that 
# PyTorch's native conv is already highly optimized via cuDNN. 
# The real opportunity here is fusing BatchNorm and Scaling into the convolution output stream 
# to avoid intermediate memory writes/reads if possible, or simply optimizing the BN+Scale part.
#
# Since replacing the entire Conv2d with a custom naive CUDA kernel would likely be slower than cuDNN,
# we will focus on fusing the BatchNorm and Scaling operations into a single efficient kernel 
# that operates on the output of the convolution. This reduces memory bandwidth pressure.
# We will keep nn.Conv2d for the heavy lifting (assuming cuDNN is available) but replace 
# the subsequent BN and Scale with a custom fused kernel.

fused_bn_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void fused_bn_scale_kernel(
    const float* input,      // Output of Conv2d: [N, C, H, W]
    const float* weight,     // BN weight (gamma)
    const float* bias,       // BN bias (beta)
    const float* running_mean,
    const float* running_var,
    float* output,           // Final output: [N, C, H, W]
    int N, int C, int H, int W,
    float eps,
    float scale_factor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * H * W;

    if (idx < total_elements) {
        // Calculate spatial and channel indices
        int w_idx = idx % W;
        int temp = idx / W;
        int h_idx = temp % H;
        temp /= H;
        int c_idx = temp % C;
        int n_idx = temp / C;

        // Linear index for the specific channel's mean/var/weight/bias
        int c_linear = c_idx;

        float val = input[idx];
        
        // Batch Normalization: (x - mean) / sqrt(var + eps)
        float inv_std = rsqrtf(running_var[c_linear] + eps);
        float normalized = (val - running_mean[c_linear]) * inv_std;
        
        // Affine transform: gamma * normalized + beta
        float affine = weight[c_linear] * normalized + bias[c_linear];
        
        // Scaling factor
        output[idx] = affine * scale_factor;
    }
}

torch::Tensor fused_bn_scale_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps,
    float scale_factor
) {
    auto N = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);
    
    auto output = torch::empty_like(input);

    const int block_size = 256;
    int total_elements = N * C * H * W;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_bn_scale_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        eps,
        scale_factor
    );

    return output;
}
"""

fused_bn_scale_cpp_source = (
    "torch::Tensor fused_bn_scale_cuda("
    "torch::Tensor input,"
    "torch::Tensor weight,"
    "torch::Tensor bias,"
    "torch::Tensor running_mean,"
    "torch::Tensor running_var,"
    "float eps,"
    "float scale_factor"
    ");"
)

# Compile the inline CUDA code
fused_bn_scale = load_inline(
    name="fused_bn_scale",
    cpp_sources=fused_bn_scale_cpp_source,
    cuda_sources=fused_bn_scale_source,
    functions=["fused_bn_scale_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution, then fuses Batch Normalization and Scaling 
    into a single custom CUDA kernel to reduce memory overhead.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # We keep the BN module structure to access its parameters easily, 
        # but we won't use its forward method. Instead, we extract its state.
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor
        self.eps = self.bn.eps

    def forward(self, x):
        # 1. Convolution (Native PyTorch, likely using cuDNN for speed)
        x = self.conv(x)
        
        # 2. Fused BatchNorm + Scaling (Custom CUDA Kernel)
        # Extract running stats and parameters from the BN layer
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        weight = self.bn.weight
        bias = self.bn.bias
        
        x = fused_bn_scale.fused_bn_scale_cuda(
            x, 
            weight, 
            bias, 
            running_mean, 
            running_var, 
            self.eps, 
            self.scaling_factor
        )
        
        return x

# Helper functions to match the interface expected by the prompt structure
def get_inputs():
    return [torch.rand(128, 8, 128, 128)]

def get_init_inputs():
    return [8, 64, 3, 2.0]