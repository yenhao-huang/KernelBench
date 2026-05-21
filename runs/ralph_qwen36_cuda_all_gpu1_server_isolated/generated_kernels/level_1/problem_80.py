import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA implementation for 2D Convolution with dilation and asymmetric padding
# This kernel handles:
# 1. Arbitrary kernel sizes (H, W)
# 2. Dilation in both height and width
# 3. Asymmetric padding (pad_h_top/bottom, pad_w_left/right)
# 4. FP32 precision

conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Helper to calculate output dimensions
__device__ int get_out_height(int in_h, int kernel_h, int pad_top, int pad_bottom, int stride, int dilation) {
    return (in_h + pad_top + pad_bottom - dilation * (kernel_h - 1) - 1) / stride + 1;
}

__device__ int get_out_width(int in_w, int kernel_w, int pad_left, int pad_right, int stride, int dilation) {
    return (in_w + pad_left + pad_right - dilation * (kernel_w - 1) - 1) / stride + 1;
}

__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias, // Can be null if no bias
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_top,
    int pad_bottom,
    int pad_left,
    int pad_right,
    int dilation_h,
    int dilation_w,
    bool has_bias
) {
    // Each thread computes one output element (batch, out_channel, out_h, out_w)
    // We use a 1D grid mapping to the flattened output space
    
    int total_out = batch_size * out_channels * get_out_height(in_height, kernel_h, pad_top, pad_bottom, stride_h, dilation_h) * 
                            get_out_width(in_width, kernel_w, pad_left, pad_right, stride_w, dilation_w);
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_out) return;

    // Decode index into coordinates
    int out_w_idx = idx % get_out_width(in_width, kernel_w, pad_left, pad_right, stride_w, dilation_w);
    int temp = idx / get_out_width(in_width, kernel_w, pad_left, pad_right, stride_w, dilation_w);
    
    int out_h_idx = temp % get_out_height(in_height, kernel_h, pad_top, pad_bottom, stride_h, dilation_h);
    temp = temp / get_out_height(in_height, kernel_h, pad_top, pad_bottom, stride_h, dilation_h);
    
    int out_c_idx = temp % out_channels;
    int b_idx = temp / out_channels;

    // Calculate the starting position in the input space corresponding to this output pixel
    // Input coordinate relative to padded image
    int start_h = out_h_idx * stride_h - pad_top;
    int start_w = out_w_idx * stride_w - pad_left;

    float sum = 0.0f;

    // Iterate over input channels and kernel elements
    for (int c = 0; c < in_channels; ++c) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                // Calculate input coordinates with dilation
                int ih = start_h + kh * dilation_h;
                int iw = start_w + kw * dilation_w;

                // Check bounds within the original input image (excluding padding area which is 0)
                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                    // Input index: [b, c, h, w]
                    int input_idx = ((b_idx * in_channels + c) * in_height + ih) * in_width + iw;
                    
                    // Weight index: [out_c, c, kh, kw]
                    int weight_idx = ((out_c_idx * in_channels + c) * kernel_h + kh) * kernel_w + kw;

                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    if (has_bias) {
        sum += bias[out_c_idx];
    }

    // Output index: [b, out_c, out_h, out_w]
    int out_h = get_out_height(in_height, kernel_h, pad_top, pad_bottom, stride_h, dilation_h);
    int out_w = get_out_width(in_width, kernel_w, pad_left, pad_right, stride_w, dilation_w);
    int output_idx = ((b_idx * out_channels + out_c_idx) * out_h + out_h_idx) * out_w + out_w_idx;
    
    output[output_idx] = sum;
}

torch::Tensor conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h,
    int stride_w,
    int pad_top,
    int pad_bottom,
    int pad_left,
    int pad_right,
    int dilation_h,
    int dilation_w
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto in_height = input.size(2);
    const auto in_width = input.size(3);
    
    const auto out_channels = weight.size(0);
    const auto kernel_h = weight.size(2);
    const auto kernel_w = weight.size(3);

    const auto out_h = get_out_height(in_height, kernel_h, pad_top, pad_bottom, stride_h, dilation_h);
    const auto out_w = get_out_width(in_width, kernel_w, pad_left, pad_right, stride_w, dilation_w);

    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, input.options());

    const int block_size = 256;
    const int total_elements = batch_size * out_channels * out_h * out_w;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    bool has_bias = !bias.numel(); // If bias is empty or not provided
    
    // Handle case where bias might be passed but empty tensor
    if (bias.numel() == 0) {
        has_bias = false;
    }

    conv2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        has_bias ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        dilation_h,
        dilation_w,
        has_bias
    );

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in conv2d: ") + cudaGetErrorString(err));
    }

    return output;
}
"""

conv2d_cpp_source = (
    "torch::Tensor conv2d_cuda("
    "torch::Tensor input,"
    "torch::Tensor weight,"
    "torch::Tensor bias,"
    "int stride_h,"
    "int stride_w,"
    "int pad_top,"
    "int pad_bottom,"
    "int pad_left,"
    "int pad_right,"
    "int dilation_h,"
    "int dilation_w"
    ");"
)

# Compile the inline CUDA code
conv2d_module = load_inline(
    name="conv2d_custom",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_source,
    functions=["conv2d_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Optimized 2D Convolution using custom CUDA operator.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Store parameters for forward pass
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_h, self.kernel_w = kernel_size
        self.stride_h = stride if isinstance(stride, int) else stride[0]
        self.stride_w = stride if isinstance(stride, int) else stride[1]
        
        # Handle asymmetric padding: (pad_h_top/bottom, pad_w_left/right)
        # If padding is a single int, it applies to all sides. 
        # The problem description says padding is tuple (top/bottom, left/right).
        if isinstance(padding, int):
            self.pad_top = self.pad_bottom = padding
            self.pad_left = self.pad_right = padding
        else:
            # padding is (pad_h, pad_w) or (pad_top/bottom, pad_left/right)?
            # Standard PyTorch nn.Conv2d padding is usually symmetric int or tuple (H, W).
            # The docstring says: "Padding applied to the input (top/bottom, left/right)".
            # This implies padding[0] applies to top and bottom, padding[1] to left and right.
            self.pad_top = padding[0]
            self.pad_bottom = padding[0]
            self.pad_left = padding[1]
            self.pad_right = padding[1]

        self.dilation_h = dilation if isinstance(dilation, int) else dilation[0]
        self.dilation_w = dilation if isinstance(dilation, int) else dilation[1]
        
        self.has_bias = bias
        
        # Initialize weights and biases manually to match nn.Conv2d behavior
        # nn.Conv2d uses Kaiming uniform initialization for weights
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_h, kernel_w))
        nn.init.kaiming_uniform_(self.weight, a=0)
        
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            self.bias = nn.Parameter(torch.empty(out_channels))
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 2D convolution.
        """
        if self.has_bias and self.bias is not None:
            bias_tensor = self.bias
        else:
            # Pass empty tensor if no bias
            bias_tensor = torch.empty(0, device=x.device)
            
        return conv2d_module.conv2d_cuda(
            x,
            self.weight,
            bias_tensor,
            self.stride_h,
            self.stride_w,
            self.pad_top,
            self.pad_bottom,
            self.pad_left,
            self.pad_right,
            self.dilation_h,
            self.dilation_w
        )

import math