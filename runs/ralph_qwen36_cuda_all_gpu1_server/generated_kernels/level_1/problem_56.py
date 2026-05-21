import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 2D convolution
# This implementation uses a naive but correct approach for general asymmetric kernels.
# For production, one would typically use cuDNN or cutlass, but here we implement a custom kernel
# to demonstrate inline CUDA integration as requested.
# We will use a simple row-major blocking strategy.

conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Helper macro for error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void conv2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int groups
) {
    // Calculate output dimensions
    int out_h = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_w = (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    // Each thread computes one element of the output tensor
    // Indexing: batch, out_channel, out_h, out_w
    
    int total_out_elements = batch_size * out_channels * out_h * out_w;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_out_elements) return;

    // Decode linear index to 4D coordinates
    int ow = idx % out_w;
    int rem = idx / out_w;
    int oh = rem % out_h;
    rem = rem / out_h;
    int oc = rem % out_channels;
    int b = rem / out_channels;

    float sum = 0.0f;
    
    // Determine the channel range for this output channel based on groups
    int group_idx = oc / (out_channels / groups);
    int ic_start = group_idx * (in_channels / groups);
    int ic_end = ic_start + (in_channels / groups);

    // Iterate over input channels, kernel height, and kernel width
    for (int ic = ic_start; ic < ic_end; ++ic) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                // Calculate input coordinates with padding and dilation
                int ih = oh * stride_h + kh * dilation_h - pad_h;
                int iw = ow * stride_w + kw * dilation_w - pad_w;

                // Check bounds
                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    // Input index: N, C, H, W
                    int input_idx = ((b * in_channels + ic) * height + ih) * width + iw;
                    
                    // Weight index: O, I, KH, KW (assuming standard Conv2d weight layout [out_c, in_c/groups, k_h, k_w])
                    // Note: PyTorch nn.Conv2d weights are [out_channels, in_channels/groups, kernel_height, kernel_width]
                    int weight_idx = ((oc * (in_channels / groups) + (ic - ic_start)) * kernel_h + kh) * kernel_w + kw;

                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Add bias if present
    if (bias != nullptr) {
        sum += bias[oc];
    }

    // Output index: N, O, H, W
    int output_idx = ((b * out_channels + oc) * out_h + oh) * out_w + ow;
    output[output_idx] = sum;
}

torch::Tensor conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    
    // Validate dimensions
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N, C, H, W)");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D (O, I/groups, KH, KW)");

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    int out_channels = weight.size(0);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    
    // Extract stride, padding, dilation from the model parameters passed via wrapper or hardcoded defaults if not exposed.
    // Since we are replacing nn.Conv2d, we need to know these params. 
    // The load_inline function doesn't easily pass complex structs without modifying the signature.
    // We will assume standard values or pass them as arguments. 
    // To make this generic for the provided Model class which has specific args, 
    // we should ideally pass stride, padding, dilation, groups to the kernel launch.
    // However, the example shows a simple function signature. 
    // Let's modify the C++ interface to accept these parameters explicitly.
    
    return torch::zeros_like(input); // Placeholder
}

// We need a more robust wrapper that accepts all conv parameters
torch::Tensor conv2d_cuda_full(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int groups
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    int out_channels = weight.size(0);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    // Calculate output dimensions
    int out_h = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_w = (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, input.options());

    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.numel() > 0 ? bias.data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();

    int total_out_elements = batch_size * out_channels * out_h * out_w;
    
    if (total_out_elements == 0) {
        return output;
    }

    const int block_size = 256;
    const int num_blocks = (total_out_elements + block_size - 1) / block_size;

    conv2d_kernel<<<num_blocks, block_size>>>(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups
    );

    CUDA_CHECK(cudaGetLastError());
    
    return output;
}
"""

conv2d_cpp_source = (
    "torch::Tensor conv2d_cuda_full("
    "torch::Tensor input,"
    "torch::Tensor weight,"
    "torch::Tensor bias,"
    "int stride_h,"
    "int stride_w,"
    "int pad_h,"
    "int pad_w,"
    "int dilation_h,"
    "int dilation_w,"
    "int groups"
    ");"
)

# Compile the inline CUDA code
conv2d_module = load_inline(
    name="conv2d_custom",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_source,
    functions=["conv2d_cuda_full"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Optimized 2D convolution using a custom CUDA operator.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Store parameters for the forward pass
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and biases using PyTorch's standard initialization 
        # to ensure compatibility with pre-trained models or correct initial state.
        # We don't store them as nn.Parameter here because the custom kernel expects raw tensors,
        # but we need to manage their lifecycle. Alternatively, we can keep them as parameters
        # and pass data_ptr in forward.
        
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming uniform initialization for weights, similar to nn.Conv2d
        nn.init.kaiming_uniform_(self.weight, a=0.05)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using the custom CUDA operator.
        """
        # Extract parameters
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding
        dilation_h, dilation_w = self.dilation
        
        # Prepare bias tensor if it exists
        bias_tensor = self.bias if self.bias is not None else torch.empty(0, device=x.device)
        
        # Call the custom CUDA kernel
        out = conv2d_module.conv2d_cuda_full(
            x,
            self.weight,
            bias_tensor,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            self.groups
        )
        
        return out

import math