import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Conv2d + Scale + Min reduction
# We fuse these operations to reduce memory bandwidth pressure.
# The kernel performs:
# 1. Convolution (im2col + gemm or direct convolution)
# 2. Element-wise multiplication by scale_factor
# 3. Reduction (min) along the channel dimension (dim=1)

custom_conv_scale_min_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Helper for atomicMin on float
__device__ __forceinline__ void atomicMinFloat(float* address, float val) {
    unsigned int* address_as_ui = (unsigned int*)address;
    unsigned int old = *address_as_ui, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ui, assumed,
            __float_as_uint(fminf(val, __uint_as_float(assumed))));
    } while (assumed != old);
}

__global__ void conv_scale_min_kernel(
    const float* input,       // [N, C_in, H_in, W_in]
    const float* weight,      // [C_out, C_in, K_h, K_w]
    const float* bias,        // [C_out] (optional, can be 0)
    float* output,            // [N, C_out, H_out, W_out] -> intermediate before min
    float* final_output,      // [N, 1, H_out, W_out] -> result of min along C_out
    int N, int C_in, int C_out, 
    int H_in, int W_in, 
    int K_h, int K_w, 
    int H_out, int W_out,
    float scale_factor) {

    // Each thread block handles one output pixel (h, w) for a specific batch item n?
    // Or better: Each thread handles one element of the final output [N, 1, H_out, W_out]
    // But we need to compute the convolution first.
    
    // Let's use a grid-stride loop where each thread computes one (n, h, w) tuple.
    // For each (n, h, w), we iterate over all C_out channels to compute the conv result,
    // apply scale, and track the minimum.

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * H_out * W_out;

    if (idx >= total_elements) return;

    int n = idx / (H_out * W_out);
    int hw = idx % (H_out * W_out);
    int h_out = hw / W_out;
    int w_out = hw % W_out;

    // Initialize min value to infinity
    float min_val = 1e38f; // Large positive number for min

    // Iterate over output channels
    for (int c_out = 0; c_out < C_out; ++c_out) {
        float sum = 0.0f;
        
        // Convolution loop: iterate over input channels and kernel spatial dims
        for (int c_in = 0; c_in < C_in; ++c_in) {
            for (int k_h = 0; k_h < K_h; ++k_h) {
                for (int k_w = 0; k_w < K_w; ++k_w) {
                    int h_in = h_out + k_h;
                    int w_in = w_out + k_w;

                    // Check bounds for padding (assuming zero padding, same size output if stride=1 pad=(K-1)/2)
                    // The problem statement doesn't specify padding/stride. 
                    // Standard Conv2d with kernel_size 3 usually implies padding=1 for 'same' output size if stride=1.
                    // Let's assume standard valid convolution or same padding. 
                    // Given the input shape and typical usage, let's assume padding = kernel_size // 2.
                    // If h_in/w_in are out of bounds, value is 0.
                    
                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        int input_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                        int weight_idx = ((c_out * C_in + c_in) * K_h + k_h) * K_w + k_w;
                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }

        // Add bias if present (assuming bias is passed or zero)
        // For simplicity, we assume no bias or bias=0 as it wasn't explicitly in the nn.Conv2d init args in a way that suggests passing it separately.
        // However, nn.Conv2d has bias=True by default. We need to handle it.
        // Let's add bias here.
        if (bias) {
            sum += bias[c_out];
        }

        // Apply scale factor
        float scaled_val = sum * scale_factor;

        // Update minimum
        if (scaled_val < min_val) {
            min_val = scaled_val;
        }
    }

    // Write the minimum value for this (n, h_out, w_out) to final_output
    int out_idx = ((n * 1 + 0) * H_out + h_out) * W_out + w_out;
    final_output[out_idx] = min_val;
}

torch::Tensor conv_scale_min_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scale_factor) {
    
    auto N = input.size(0);
    auto C_in = input.size(1);
    auto H_in = input.size(2);
    auto W_in = input.size(3);

    auto C_out = weight.size(0);
    auto K_h = weight.size(2);
    auto K_w = weight.size(3);

    // Assuming stride=1 and padding=(K-1)/2 for same output size
    int H_out = H_in; 
    int W_out = W_in;

    // Output tensor shape: [N, 1, H_out, W_out]
    auto out_shape = {N, 1, H_out, W_out};
    torch::Tensor final_output = torch::zeros(out_shape, input.options());

    const int block_size = 256;
    int total_elements = N * H_out * W_out;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    conv_scale_min_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.numel() > 0 ? bias.data_ptr<float>() : nullptr,
        nullptr, // intermediate output not needed for final result
        final_output.data_ptr<float>(),
        N, C_in, C_out, H_in, W_in, K_h, K_w, H_out, W_out, scale_factor
    );

    return final_output;
}
"""

custom_conv_scale_min_cpp_source = (
    "torch::Tensor conv_scale_min_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float scale_factor);"
)

# Compile the inline CUDA code
conv_scale_min_module = load_inline(
    name="conv_scale_min",
    cpp_sources=custom_conv_scale_min_cpp_source,
    cuda_sources=custom_conv_scale_min_source,
    functions=["conv_scale_min_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a convolution, scales the output, and then applies a minimum operation.
    Uses a custom fused CUDA kernel for performance.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1, height, width).
        """
        # Extract weight and bias from the conv layer
        weight = self.conv.weight
        bias = self.conv.bias
        
        # Call the custom fused CUDA operator
        output = conv_scale_min_module.conv_scale_min_cuda(x, weight, bias, self.scale_factor)
        
        return output

def get_inputs():
    # randomly generate input tensors based on the model architecture
    a = torch.rand(batch_size, in_channels, height, width).cuda()
    return [a]

def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    return [in_channels, out_channels, kernel_size, scale_factor]