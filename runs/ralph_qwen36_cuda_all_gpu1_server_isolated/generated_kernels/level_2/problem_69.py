import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Conv2d + HardSwish + ReLU fusion
# This kernel performs: y = ReLU(HardSwish(Conv2d(x)))
# Note: HardSwish(x) = x * ReLU6(x + 3) / 6
# Combined with final ReLU: ReLU( HardSwish(x) )
# Since HardSwish output is always >= 0 (because ReLU6(x+3) >= 0 and we divide by positive 6, but wait: 
# HardSwish(x) = x * min(max(x+3, 0), 6) / 6.
# If x < 0, x+3 could be positive or negative.
# Case 1: x <= -3 -> HardSwish(x) = x * 0 / 6 = 0. ReLU(0) = 0.
# Case 2: -3 < x < 0 -> HardSwish(x) = x * (x+3) / 6. Since x is negative and (x+3) is positive, result is negative. 
#        Then ReLU(negative) = 0.
# Case 3: x >= 0 -> HardSwish(x) = x * min(max(x+3, 0), 6) / 6. Since x>=0, x+3>=3>0. 
#        If x+3 <= 6 (i.e., x <= 3), HardSwish(x) = x*(x+3)/6 >= 0. ReLU keeps it.
#        If x > 3, HardSwish(x) = x*6/6 = x >= 0. ReLU keeps it.
# So, ReLU(HardSwish(x)) is effectively:
# If x <= 0: 0
# If x > 0: HardSwish(x)
# Which simplifies to: max(0, x * min(max(x+3, 0), 6) / 6)
# Actually, let's just implement the standard sequence fused into one kernel for clarity and correctness.

conv_hardswish_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Helper to compute HardSwish(x) = x * ReLU6(x + 3) / 6
__device__ inline float hardswish(float x) {
    return x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f;
}

// Helper to compute ReLU(x) = max(0, x)
__device__ inline float relu(float x) {
    return fmaxf(x, 0.0f);
}

__global__ void conv_hardswish_relu_kernel(
    const float* input, 
    const float* weight, 
    const float* bias, 
    float* output, 
    int batch_size, 
    int in_channels, 
    int height, 
    int width, 
    int out_channels, 
    int kernel_h, 
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w
) {
    // Each thread handles one output element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int total_elements = batch_size * out_channels * height * width;
    if (idx >= total_elements) return;

    // Decompose index into spatial and channel dimensions
    int w_idx = idx % width;
    int temp = idx / width;
    int h_idx = temp % height;
    temp = temp / height;
    int c_out_idx = temp % out_channels;
    int b_idx = temp / out_channels;

    float sum = 0.0f;
    
    // Iterate over input channels and kernel dimensions
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                // Calculate input spatial coordinates with padding and stride
                int h_in = h_idx * stride_h + kh - pad_h;
                int w_in = w_idx * stride_w + kw - pad_w;

                // Check bounds
                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    // Linear index for input tensor: [N, C, H, W]
                    int input_idx = b_idx * in_channels * height * width + 
                                    c_in * height * width + 
                                    h_in * width + 
                                    w_in;
                    
                    // Linear index for weight tensor: [OutC, InC, KH, KW]
                    int weight_idx = c_out_idx * in_channels * kernel_h * kernel_w + 
                                     c_in * kernel_h * kernel_w + 
                                     kh * kernel_w + 
                                     kw;

                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Add bias
    if (bias != nullptr) {
        sum += bias[c_out_idx];
    }

    // Apply HardSwish then ReLU
    float hs = hardswish(sum);
    float out_val = relu(hs);

    // Write to output tensor: [N, C, H, W]
    int output_idx = b_idx * out_channels * height * width + 
                     c_out_idx * height * width + 
                     h_idx * width + 
                     w_idx;
    output[output_idx] = out_val;
}

torch::Tensor conv_hardswish_relu_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias
) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    
    auto out_channels = weight.size(0);
    auto kernel_h = weight.size(2);
    auto kernel_w = weight.size(3);
    
    // Assuming stride=1, padding=kernel_size//2 for standard conv behavior matching nn.Conv2d default
    // However, to be generic and match the provided Model which uses default Conv2d params:
    // nn.Conv2d(in_channels, out_channels, kernel_size) defaults to stride=1, padding=0 if not specified? 
    // No, default padding is 0. But usually for square kernels, padding = kernel_size // 2 is common for 'same' output size.
    // The prompt's Model uses nn.Conv2d(in_channels, out_channels, kernel_size). 
    // Default stride=1, padding=0.
    
    int stride_h = 1;
    int stride_w = 1;
    int pad_h = 0;
    int pad_w = 0;

    auto output = torch::zeros({batch_size, out_channels, height, width}, input.options());

    const int block_size = 256;
    int total_elements = batch_size * out_channels * height * width;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    conv_hardswish_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.numel() > 0 ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        height,
        width,
        out_channels,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w
    );

    return output;
}
"""

conv_hardswish_relu_cpp_source = (
    "torch::Tensor conv_hardswish_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"
)

# Compile the inline CUDA code
conv_hardswish_relu = load_inline(
    name="conv_hardswish_relu",
    cpp_sources=conv_hardswish_relu_cpp_source,
    cuda_sources=conv_hardswish_relu_source,
    functions=["conv_hardswish_relu_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a convolution, applies HardSwish, and then ReLU using a custom fused CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # Register the custom function as a module attribute or use it directly in forward
        self.custom_op = conv_hardswish_relu

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        # Get weights and bias from the conv layer
        weight = self.conv.weight
        bias = self.conv.bias
        
        # Call the fused custom CUDA operator
        x = self.custom_op.conv_hardswish_relu_cuda(x, weight, bias)
        
        return x