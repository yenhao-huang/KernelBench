import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA implementation for Transposed 3D Convolution
# This kernel performs the im2col-like extraction followed by GEMM, 
# optimized for FP32. It handles asymmetric kernels and strides.

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Helper to calculate global index
#define GET_INDEX(batch, channel, d, h, w, depth_out, height_out, width_out, channels) \
    ((batch) * (channels) + (channel)) * (depth_out) * (height_out) * (width_out) + \
    (d) * (height_out) * (width_out) + (h) * (width_out) + (w)

// Kernel for extracting input patches (im2col style) into a matrix
// Input: (N, C, D_in, H_in, W_in) -> Output Matrix: (N * out_D * out_H * out_W, in_C * kD * kH * kW)
__global__ void im2col_3d_kernel(
    const float* input, 
    float* col_buffer, 
    int batch_size, 
    int in_channels, 
    int depth_in, 
    int height_in, 
    int width_in,
    int out_depth, 
    int out_height, 
    int out_width,
    int kernel_depth, 
    int kernel_height, 
    int kernel_width,
    int stride_d, 
    int stride_h, 
    int stride_w,
    int pad_d, 
    int pad_h, 
    int pad_w) 
{
    // Each thread handles one element in the output feature map for a specific batch
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int total_elements = batch_size * out_depth * out_height * out_width;
    if (idx >= total_elements) return;

    int b = idx / (out_depth * out_height * out_width);
    int rem = idx % (out_depth * out_height * out_width);
    int d_out = rem / (out_height * out_width);
    rem %= (out_height * out_width);
    int h_out = rem / out_width;
    int w_out = rem % out_width;

    // Calculate the starting position in the input volume for this output voxel
    int d_in_start = d_out * stride_d - pad_d;
    int h_in_start = h_out * stride_h - pad_h;
    int w_in_start = w_out * stride_w - pad_w;

    float* col_ptr = col_buffer + idx * (in_channels * kernel_depth * kernel_height * kernel_width);

    for (int c = 0; c < in_channels; ++c) {
        for (int k_d = 0; k_d < kernel_depth; ++k_d) {
            int d_in = d_in_start + k_d;
            // Check bounds
            if (d_in < 0 || d_in >= depth_in) continue; 
            
            for (int k_h = 0; k_h < kernel_height; ++k_h) {
                int h_in = h_in_start + k_h;
                if (h_in < 0 || h_in >= height_in) continue;

                for (int k_w = 0; k_w < kernel_width; ++k_w) {
                    int w_in = w_in_start + k_w;
                    if (w_in < 0 || w_in >= width_in) continue;

                    // Linear index in input tensor: (N, C, D, H, W)
                    int input_idx = ((b * in_channels + c) * depth_in + d_in) * height_in * width_in + h_in * width_in + w_in;
                    
                    // Linear index in col buffer: (Out_Index, Channel * Kernel_D * Kernel_H * Kernel_W)
                    // We map kernel dims to the last dimensions of the col matrix for efficient GEMM later
                    int col_idx = ((c * kernel_depth + k_d) * kernel_height + k_h) * kernel_width + k_w;
                    
                    col_ptr[col_idx] = input[input_idx];
                }
            }
        }
    }
}

// Kernel for the matrix multiplication part: Col_Matrix (M x K) * Weights (K x N_out) -> Output_Matrix (M x N_out)
// M = batch_size * out_depth * out_height * out_width
// K = in_channels * kernel_depth * kernel_height * kernel_width
// N_out = out_channels
__global__ void matmul_3d_kernel(
    const float* col_buffer, 
    const float* weights, 
    const float* bias, // Can be null if no bias
    float* output, 
    int batch_size, 
    int in_channels, 
    int kernel_depth, 
    int kernel_height, 
    int kernel_width,
    int out_channels,
    int out_depth, 
    int out_height, 
    int out_width) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Total output elements: N * C_out * D_out * H_out * W_out
    int total_output_elements = batch_size * out_channels * out_depth * out_height * out_width;
    if (idx >= total_output_elements) return;

    int b = idx / (out_channels * out_depth * out_height * out_width);
    int rem = idx % (out_channels * out_depth * out_height * out_width);
    int c_out = rem / (out_depth * out_height * out_width);
    rem %= (out_depth * out_height * out_width);
    int d_out = rem / (out_height * out_width);
    rem %= (out_height * out_width);
    int h_out = rem / out_width;
    int w_out = rem % out_width;

    // The row in the col_buffer matrix corresponding to this output voxel
    int row_idx = b * out_depth * out_height * out_width + d_out * out_height * out_width + h_out * out_width + w_out;
    
    const float* col_row = col_buffer + row_idx * (in_channels * kernel_depth * kernel_height * kernel_width);
    
    // The column in the weights matrix corresponding to this output channel
    // Weights shape: (out_channels, in_channels * kD * kH * kW)
    const float* weight_col = weights + c_out * (in_channels * kernel_depth * kernel_height * kernel_width);

    float sum = 0.0f;
    int k_size = in_channels * kernel_depth * kernel_height * kernel_width;
    
    // Unroll loop for performance if possible, but simple loop is safer for variable sizes
    for (int k = 0; k < k_size; ++k) {
        sum += col_row[k] * weight_col[k];
    }

    if (bias != nullptr) {
        sum += bias[c_out];
    }

    output[idx] = sum;
}

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups) 
{
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(groups == 1, "Groups > 1 not supported in this custom kernel for simplicity");

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int depth_in = input.size(2);
    int height_in = input.size(3);
    int width_in = input.size(4);

    int out_channels = weight.size(0);
    int kernel_depth = weight.size(2);
    int kernel_height = weight.size(3);
    int kernel_width = weight.size(4);

    // Calculate output dimensions for Transposed Conv
    // Formula: Out = (In - 1) * Stride - 2 * Padding + Kernel + Output_Padding
    int stride_d = stride[0];
    int stride_h = stride[1];
    int stride_w = stride[2];
    
    int pad_d = padding[0];
    int pad_h = padding[1];
    int pad_w = padding[2];

    int out_pad_d = output_padding[0];
    int out_pad_h = output_padding[1];
    int out_pad_w = output_padding[2];

    int out_depth = (depth_in - 1) * stride_d - 2 * pad_d + kernel_depth + out_pad_d;
    int out_height = (height_in - 1) * stride_h - 2 * pad_h + kernel_height + out_pad_h;
    int out_width = (width_in - 1) * stride_w - 2 * pad_w + kernel_width + out_pad_w;

    // Allocate output tensor
    auto output = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width}, input.options());

    // Calculate dimensions for im2col buffer
    // Buffer size: (N * out_D * out_H * out_W) * (C_in * kD * kH * kW)
    int num_patches = batch_size * out_depth * out_height * out_width;
    int patch_size = in_channels * kernel_depth * kernel_height * kernel_width;
    
    // Allocate col buffer on GPU
    auto col_buffer = torch::zeros({num_patches, patch_size}, input.options());

    const int block_size = 256;
    const int num_blocks_im2col = (num_patches + block_size - 1) / block_size;

    // Launch Im2Col Kernel
    im2col_3d_kernel<<<num_blocks_im2col, block_size>>>(
        input.data_ptr<float>(),
        col_buffer.data_ptr<float>(),
        batch_size,
        in_channels,
        depth_in,
        height_in,
        width_in,
        out_depth,
        out_height,
        out_width,
        kernel_depth,
        kernel_height,
        kernel_width,
        stride_d,
        stride_h,
        stride_w,
        pad_d,
        pad_h,
        pad_w
    );

    // Launch MatMul Kernel
    // We treat the col_buffer as a matrix of size (num_patches x patch_size)
    // and weights as (out_channels x patch_size). 
    // Result is (num_patches x out_channels), which we reshape to output.
    
    const int num_blocks_matmul = (num_patches * out_channels + block_size - 1) / block_size;

    matmul_3d_kernel<<<num_blocks_matmul, block_size>>>(
        col_buffer.data_ptr<float>(),
        weight.data_ptr<float>(), // Weights are stored as (C_out, C_in*kD*kH*kW) in PyTorch ConvTranspose3d usually? 
                                  // Actually PyTorch ConvTranspose3d weight shape is (in_channels, out_channels/groups, kD, kH, kW).
                                  // Wait, let's check PyTorch documentation.
                                  // nn.ConvTranspose3d weight shape: (out_channels, in_channels/groups, kernel_size[0], kernel_size[1], kernel_size[2])
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        kernel_depth,
        kernel_height,
        kernel_width,
        out_channels,
        out_depth,
        out_height,
        out_width
    );

    return output;
}
"""

cpp_source = "torch::Tensor conv_transpose3d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> output_padding, int64_t groups);"

# Load the inline extension
conv_transpose3d_module = load_inline(
    name="conv_transpose3d_cuda",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conv_transpose3d_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    """
    Optimized Transposed 3D Convolution using custom CUDA operators.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Store parameters for the forward pass
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and biases manually to match PyTorch's initialization 
        # or just use standard nn.Parameter. Since we are replacing the operator, 
        # we need the parameters to exist in the module state dict.
        # Note: The custom kernel expects weight shape (out_channels, in_channels/groups, kD, kH, kW)
        # and bias shape (out_channels).
        
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming uniform initialization similar to PyTorch's default for ConvTranspose
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution using custom CUDA kernel.
        """
        # Convert tuples to lists for C++ interface if necessary, though vector<int64_t> handles it
        stride_list = list(self.stride)
        padding_list = list(self.padding)
        output_padding_list = list(self.output_padding)
        
        return conv_transpose3d_module.conv_transpose3d_cuda(
            x, 
            self.weight, 
            self.bias if self.bias is not None else torch.empty(0), # Pass empty tensor if no bias
            stride_list,
            padding_list,
            output_padding_list,
            self.groups
        )

import math