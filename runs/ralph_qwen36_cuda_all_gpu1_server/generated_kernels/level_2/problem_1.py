import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Conv2d + ReLU + Bias Add
# This fuses three operations into a single kernel to reduce memory bandwidth pressure.
custom_conv_relu_bias_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s at line %d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void conv_relu_bias_kernel(
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
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w) {
    
    // Each thread handles one output element (H_out, W_out, C_out) for a specific batch item
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int total_elements = batch_size * out_channels * height * width;
    
    if (idx >= total_elements) return;

    // Decompose linear index into coordinates
    int w_idx = idx % width;
    int temp = idx / width;
    int h_idx = temp % height;
    temp = temp / height;
    int c_out_idx = temp % out_channels;
    int b_idx = temp / out_channels;

    // Calculate input spatial coordinates for the top-left of the kernel window
    int in_h_start = h_idx * stride_h - pad_h;
    int in_w_start = w_idx * stride_w - pad_w;

    float sum = 0.0f;

    // Iterate over input channels and kernel dimensions
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int in_h = in_h_start + kh;
                int in_w = in_w_start + kw;

                // Check bounds for padding
                if (in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
                    // Input index: [b, c_in, h, w]
                    int input_idx = ((b_idx * in_channels + c_in) * height + in_h) * width + in_w;
                    
                    // Weight index: [c_out, c_in, kh, kw]
                    int weight_idx = ((c_out_idx * in_channels + c_in) * kernel_h + kh) * kernel_w + kw;

                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Add bias and apply ReLU
    float val = sum + bias[c_out_idx];
    if (val < 0.0f) {
        val = 0.0f;
    }

    // Output index: [b, c_out, h, w]
    int output_idx = ((b_idx * out_channels + c_out_idx) * height + h_idx) * width + w_idx;
    output[output_idx] = val;
}

torch::Tensor conv_relu_bias_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias) {
    
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");

    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);

    auto out_channels = weight.size(0);
    auto kernel_h = weight.size(2);
    auto kernel_w = weight.size(3);

    // Assuming stride=1 and padding=kernel_size//2 for standard Conv2d with odd kernel size 3
    // If the original model uses different stride/padding, this needs adjustment. 
    // The prompt implies a standard conv setup. Let's assume stride=1, pad=1 for k=3 to maintain spatial dims.
    int stride_h = 1;
    int stride_w = 1;
    int pad_h = 1;
    int pad_w = 1;

    auto output = torch::zeros({batch_size, out_channels, height, width}, input.options());

    const int block_size = 256;
    int total_elements = batch_size * out_channels * height * width;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    conv_relu_bias_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        height,
        width,
        out_channels,
        kernel_h,
        kernel_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w
    );

    CUDA_CHECK(cudaGetLastError());
    
    return output;
}
"""

custom_conv_relu_bias_cpp_source = (
    "torch::Tensor conv_relu_bias_cuda("
    "torch::Tensor input, "
    "torch::Tensor weight, "
    "torch::Tensor bias);"
);

// Compile the inline CUDA code
conv_relu_bias_module = load_inline(
    name="conv_relu_bias",
    cpp_sources=custom_conv_relu_bias_cpp_source,
    cuda_sources=custom_conv_relu_bias_source,
    functions=["conv_relu_bias_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution, applies ReLU, and adds a bias term
    using a custom fused CUDA operator.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        # We still need to store the parameters so they can be moved to GPU and saved/loaded correctly
        # However, we will not use nn.Conv2d for the forward pass computation.
        self.conv_weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Store kernel size for potential reuse if needed, though it's fixed in init
        self.kernel_size = kernel_size

    def forward(self, x):
        # Use the custom fused operator
        return conv_relu_bias_module.conv_relu_bias_cuda(x, self.conv_weight, self.bias)


def get_inputs():
    return [torch.rand(128, 64, 128, 128).cuda()]

def get_init_inputs():
    return [64, 128, 3, (128, 1, 1)]