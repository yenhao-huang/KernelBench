import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Conv2d + Subtraction + Mish activation
# This fuses the operations to reduce memory bandwidth pressure.
# We assume input is NHWC or NCHW. The original model uses NCHW (Conv2d default).
# To optimize, we process in NCHW layout.

custom_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Helper for Mish activation: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
__device__ inline float mish(float x) {
    return x * tanhf(logf(1.0f + expf(x)));
}

__global__ void conv_sub_mish_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, // Conv2d usually has bias, though not explicitly used in subtraction logic, we need to handle it if present. 
                                    // Note: nn.Conv2d has bias=True by default. We must account for it.
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int height,
    int width,
    int out_channels,
    int kernel_h,
    int kernel_w,
    float sub_val_1,
    float sub_val_2
) {
    // Each thread handles one output element (N, C_out, H_out, W_out)
    int n = blockIdx.z;
    int c_out = blockIdx.y;
    int h_out = threadIdx.x / gridDim.x; // This mapping is tricky for 3D blocks. Let's use standard 1D block mapping for simplicity or 2D/3D grid.
    
    // Better mapping: 1D grid of threads, each thread computes one output pixel
    int total_elements = batch_size * out_channels * height * width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_elements) return;
    
    // Decode index to coordinates
    int w_out = idx % height; // Wait, standard is NCHW. Let's stick to NCHW indexing.
    // Index in NCHW: n * (C_out * H * W) + c_out * (H * W) + h_out * W + w_out
    
    int hw = height * width;
    int cw = out_channels * hw;
    
    int w = idx % width;
    int temp = idx / width;
    int h = temp % height;
    int c = temp / height;
    int n = temp / (height * out_channels); // This logic is flawed if we just use linear index.
    
    // Let's re-decode properly for NCHW layout:
    // idx = n * (C_out * H * W) + c_out * (H * W) + h * W + w
    
    int w_local = idx % width;
    int rem = idx / width;
    int h_local = rem % height;
    rem = rem / height;
    int c_local = rem % out_channels;
    int n_local = rem / out_channels;
    
    // Calculate input pointer offset for this output element
    // Input shape: N, C_in, H, W
    // Output shape: N, C_out, H, W
    
    float sum = 0.0f;
    
    // Convolution loop
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h_in = h_local + kh - kernel_h / 2;
                int w_in = w_local + kw - kernel_w / 2;
                
                // Handle padding (assuming 'same' padding or similar, but nn.Conv2d with default padding=0 needs boundary checks)
                // The original code uses nn.Conv2d(in_channels, out_channels, kernel_size). Default padding is 0.
                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    int input_idx = n_local * (in_channels * hw) + c_in * hw + h_in * width + w_in;
                    int weight_idx = c_out * (in_channels * kernel_h * kernel_w) + c_in * (kernel_h * kernel_w) + kh * kernel_w + kw;
                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Add bias if it exists. 
    // Note: The custom kernel signature above didn't include bias pointer explicitly in the simplified view, 
    // but nn.Conv2d has a bias tensor. We need to pass it or assume it's handled.
    // To keep the kernel simple and robust without modifying the Python wrapper too much, 
    // we can assume bias is zero if not passed, OR we modify the signature.
    // Let's add bias pointer to the signature.
    
    // Re-defining kernel with bias support in the actual implementation below.
}

// Optimized Kernel with Bias Support
__global__ void conv_sub_mish_kernel_v2(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int height,
    int width,
    int out_channels,
    int kernel_h,
    int kernel_w,
    float sub_val_1,
    float sub_val_2
) {
    int total_elements = batch_size * out_channels * height * width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_elements) return;
    
    int w_local = idx % width;
    int rem = idx / width;
    int h_local = rem % height;
    rem = rem / height;
    int c_local = rem % out_channels;
    int n_local = rem / out_channels;
    
    float sum = 0.0f;
    
    // Convolution
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h_in = h_local + kh - kernel_h / 2;
                int w_in = w_local + kw - kernel_w / 2;
                
                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    int input_idx = n_local * (in_channels * height * width) + c_in * (height * width) + h_in * width + w_in;
                    int weight_idx = c_out * (in_channels * kernel_h * kernel_w) + c_in * (kernel_h * kernel_w) + kh * kernel_w + kw;
                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Add bias
    if (bias != nullptr) {
        sum += bias[c_local];
    }
    
    // Subtract values
    sum -= sub_val_1;
    sum -= sub_val_2;
    
    // Apply Mish activation
    output[idx] = mish(sum);
}

torch::Tensor conv_sub_mish_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias,
    float sub_val_1,
    float sub_val_2
) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    
    auto out_channels = weight.size(0);
    auto kernel_h = weight.size(2);
    auto kernel_w = weight.size(3);
    
    // Output shape: N, C_out, H, W
    auto output = torch::zeros({batch_size, out_channels, height, width}, input.options());
    
    const int block_size = 256;
    const int total_elements = batch_size * out_channels * height * width;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    // Ensure bias is not null if it exists, otherwise pass nullptr
    float* bias_ptr = nullptr;
    if (bias.numel() > 0) {
        bias_ptr = bias.data_ptr<float>();
    }

    conv_sub_mish_kernel_v2<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        height,
        width,
        out_channels,
        kernel_h,
        kernel_w,
        sub_val_1,
        sub_val_2
    );
    
    return output;
}
"""

custom_ops_cpp_source = (
    "torch::Tensor conv_sub_mish_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float sub_val_1, float sub_val_2);"
)

# Compile the inline CUDA code
custom_ops = load_inline(
    name="conv_sub_mish",
    cpp_sources=custom_ops_cpp_source,
    cuda_sources=custom_ops_source,
    functions=["conv_sub_mish_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a convolution, subtracts two values, applies Mish activation
    using a custom fused CUDA operator.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        # We still need the Conv2d layer to get the weights and bias initialized correctly
        # or we can initialize them manually. Using nn.Conv2d is easier for initialization.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
        
    def forward(self, x):
        # Extract weights and bias from the conv layer
        weight = self.conv.weight
        bias = self.conv.bias
        
        # Call the custom fused CUDA operator
        out = custom_ops.conv_sub_mish_cuda(
            x, 
            weight, 
            bias,
            self.subtract_value_1,
            self.subtract_value_2
        )
        return out

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2]