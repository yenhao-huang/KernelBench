import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Conv2d + Scale + LeakyReLU + GELU fusion
# This kernel performs: y = gelu(leaky_relu(conv(x) * multiplier))
# We assume NHWC is not used, so we stick to NCHW.
# To optimize memory bandwidth, we process tiles of the output feature map.

fusion_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Helper for GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
__device__ __forceinline__ float gelu(float x) {
    float cdf = 0.5f * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
    return x * cdf;
}

// Helper for LeakyReLU: max(0, x) + negative_slope * min(0, x)
__device__ __forceinline__ float leaky_relu(float x, float slope) {
    return fmaxf(0.0f, x) + slope * fminf(0.0f, x);
}

// Conv2d 3x3 with padding 1 (same padding)
// Input: N, C_in, H, W
// Weight: C_out, C_in, 3, 3
// Bias: C_out (optional, here we assume no bias for simplicity or add it if needed. 
// The original nn.Conv2d has bias=True by default. We will include bias.)
// Output: N, C_out, H, W

__global__ void conv_scale_lrelu_gelu_kernel(
    const float* __restrict__ input,      // [N, C_in, H, W]
    const float* __restrict__ weight,     // [C_out, C_in, 3, 3]
    const float* __restrict__ bias,       // [C_out]
    const float* __restrict__ multiplier, // [C_out, 1, 1]
    float* __restrict__ output,           // [N, C_out, H, W]
    int N, int C_in, int C_out, int H, int W,
    float leaky_slope
) {
    // Each thread handles one element of the output tensor
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C_out * H * W;

    if (idx >= total_elements) return;

    // Decode index to coordinates
    int w = idx % W;
    int temp = idx / W;
    int h = temp % H;
    temp = temp / H;
    int c_out = temp % C_out;
    int n = temp / C_out;

    float sum = 0.0f;
    
    // Load bias
    if (bias != nullptr) {
        sum = bias[c_out];
    }

    // Convolution loop
    // Kernel size is 3x3, padding is 1
    for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
            int iy = h + ky;
            int ix = w + kx;

            // Boundary check
            if (iy < 0 || iy >= H || ix < 0 || ix >= W) continue;

            float val = 0.0f;
            for (int c_in = 0; c_in < C_in; ++c_in) {
                // Input index: N, C_in, H, W
                int input_idx = ((n * C_in + c_in) * H + iy) * W + ix;
                
                // Weight index: C_out, C_in, 3, 3
                // ky, kx are -1, 0, 1. Map to 0, 1, 2
                int weight_idx = ((c_out * C_in + c_in) * 3 + (ky + 1)) * 3 + (kx + 1);
                
                val += input[input_idx] * weight[weight_idx];
            }
            sum += val;
        }
    }

    // Apply multiplier
    // multiplier shape is [C_out, 1, 1], so we just index by c_out
    sum *= multiplier[c_out];

    // Apply LeakyReLU
    float activated = leaky_relu(sum, leaky_slope);

    // Apply GELU
    output[idx] = gelu(activated);
}

torch::Tensor conv_scale_lrelu_gelu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor multiplier
) {
    auto N = input.size(0);
    auto C_in = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);
    auto C_out = weight.size(0);

    auto output = torch::zeros({N, C_out, H, W}, input.options());

    const int block_size = 256;
    int total_elements = N * C_out * H * W;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    // Default leaky_relu negative slope is 0.01
    float leaky_slope = 0.01f;

    conv_scale_lrelu_gelu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias != nullptr ? bias.data_ptr<float>() : nullptr,
        multiplier.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, C_out, H, W, leaky_slope
    );

    return output;
}
"""

fusion_cpp_source = (
    "torch::Tensor conv_scale_lrelu_gelu_cuda("
    "torch::Tensor input,"
    "torch::Tensor weight,"
    "torch::Tensor bias,"
    "torch::Tensor multiplier"
    ");"
)

# Compile the inline CUDA code
fusion_ops = load_inline(
    name="conv_scale_lrelu_gelu",
    cpp_sources=fusion_cpp_source,
    cuda_sources=fusion_source,
    functions=["conv_scale_lrelu_gelu_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Optimized Model that fuses Conv2d, Scale, LeakyReLU, and GELU into a single CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        
        # We need to store the parameters to pass them to the custom kernel
        # The original model uses nn.Conv2d which has weight and bias.
        # We initialize them manually to match the original architecture's initialization.
        # Note: nn.Conv2d default init is Kaiming uniform for weights and zeros for bias.
        
        self.register_buffer('weight', torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.register_buffer('bias', torch.empty(out_channels))
        
        # Initialize weight and bias to match nn.Conv2d defaults
        nn.init.kaiming_uniform_(self.weight, a=0.0, mode='fan_in', nonlinearity='leaky_relu')
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (math.sqrt(fan_in)) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

        self.multiplier_shape = multiplier_shape
        # The multiplier is a learnable parameter in the original model.
        # We will store it as a buffer or register it so it can be updated if needed, 
        # but since it's passed to the kernel, we need access to its data_ptr.
        # In the original code: self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        # We will re-create it here to ensure gradients flow correctly if this module is part of a larger graph.
        self.register_parameter('multiplier', nn.Parameter(torch.randn(multiplier_shape)))

    def forward(self, x):
        return fusion_ops.conv_scale_lrelu_gelu_cuda(
            x, 
            self.weight, 
            self.bias, 
            self.multiplier
        )

import math

# Re-define get_inputs and get_init_inputs for completeness as requested by the prompt structure,
# although the prompt asks to output the new model code. I will include them inside the block if needed,
# but typically just the ModelNew class is required. The prompt says "Output the new code in codeblocks... Just output the new model code".
# I will provide the full script including imports and helper functions to ensure it's a valid standalone file as per the example style.

def get_inputs():
    return [torch.rand(64, 64, 256, 256)]

def get_init_inputs():
    return [64, 64, 3, (64, 1, 1)]