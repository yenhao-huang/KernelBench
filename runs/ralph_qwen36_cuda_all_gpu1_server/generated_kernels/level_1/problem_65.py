import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA implementation for Transposed 2D Convolution
# This kernel performs the im2col-like extraction and matrix multiplication (GEMM) 
# or direct accumulation. For small kernels like 3x7, a direct accumulation kernel 
# is often more efficient than GEMM due to memory bandwidth constraints and overhead.
# We will implement a specialized kernel that handles the asymmetric kernel size efficiently.

transposed_conv2d_source = """
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

// Kernel for Transposed 2D Convolution with asymmetric kernel
// Assumes NHWC layout internally for easier indexing, but inputs/outputs are NCHW.
// To optimize, we process tiles of the output image.
__global__ void transposed_conv2d_kernel(
    const float* __restrict__ input,      // [N, C_in, H_in, W_in]
    const float* __restrict__ weight,     // [C_out, C_in / groups, KH, KW]
    const float* __restrict__ bias,       // [C_out] or nullptr
    float* __restrict__ output,           // [N, C_out, H_out, W_out]
    int N, int C_in, int H_in, int W_in,
    int C_out, int KH, int KW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int out_pad_h, int out_pad_w,
    int groups)
{
    // Each thread block handles a tile of the output image.
    // We use a 2D grid for N and C_out, and 2D block for H_out and W_out tiles.
    
    const int n = blockIdx.x;
    const int c_out = blockIdx.y;
    
    if (n >= N || c_out >= C_out) return;

    // Determine the input channel index this output channel corresponds to
    // For grouped convolutions, each group processes a subset of channels.
    const int group_id = c_out / (C_out / groups);
    const int c_in_base = group_id * (C_in / groups);
    
    // Shared memory for weight loading if kernel is small enough, 
    // but for 3x7=21 weights per input channel, it might be better to just load from global.
    // However, since we iterate over C_in, we can optimize access patterns.

    // Output dimensions calculation
    int H_out = (H_in - 1) * stride_h - 2 * pad_h + KH + out_pad_h;
    int W_out = (W_in - 1) * stride_w - 2 * pad_w + KW + out_pad_w;

    // Thread indices within the block for processing output pixels
    const int th = threadIdx.y;
    const int tw = threadIdx.x;
    
    // Block dimensions for output tile
    const int block_h = blockDim.y;
    const int block_w = blockDim.x;

    // Calculate global output coordinates
    int out_y = blockIdx.z * block_h + th;
    int out_x = blockIdx.z * block_w + tw; // Wait, we need a 3D grid or different mapping.
    
    // Let's use a simpler mapping: Grid(N, C_out, H_out), Block(W_out) is too large for W=512.
    // Better: Grid(N, C_out, H_out/8), Block(8, W_out/8). 
    // Let's stick to standard 3D grid: blockIdx.x=N, blockIdx.y=C_out, blockIdx.z=H_out_tile.
    // But we need to handle W_out as well.
    
    // Revised Grid/Block strategy:
    // Grid: (N, C_out, ceil(H_out/block_h))
    // Block: (block_w, block_h) -> covers a tile of output pixels.
    // We need to map threadIdx.x/y to out_x/out_y.
    
    int h_tile = blockIdx.z;
    int start_h = h_tile * blockDim.y;
    int start_w = 0; // We process all W in the block? No, block size limit is 1024.
    
    // Let's use a 2D block for (W_out, H_out) and 3D grid for (N, C_out, Tile_H)? 
    // Standard approach: Grid(N, C_out, H_out), Block(W_out). If W_out > 1024, this fails.
    // Given W=512, we can use Block(32, 16) = 512 threads.
    
    // Let's define a fixed block size for simplicity and robustness.
    // We will launch with grid(N, C_out, ceil(H_out/16), ceil(W_out/32))
    // And block(32, 16).
    
    int h_tile_idx = blockIdx.z;
    int w_tile_idx = blockIdx.w;
    
    int base_h = h_tile_idx * blockDim.y;
    int base_w = w_tile_idx * blockDim.x;
    
    int out_y_global = base_h + threadIdx.y;
    int out_x_global = base_w + threadIdx.x;

    if (out_y_global >= H_out || out_x_global >= W_out) return;

    // Calculate the corresponding input coordinates for this output pixel
    // The relationship is: out_y = in_y * stride_h - pad_h + kh_offset
    // So, in_y = (out_y + pad_h - kh_offset) / stride_h
    // We iterate over kernel positions (kh, kw) and accumulate contributions from input.
    
    float sum = 0.0f;
    
    // Iterate over kernel height and width
    for (int kh = 0; kh < KH; ++kh) {
        for (int kw = 0; kw < KW; ++kw) {
            // Calculate the input coordinate that contributes to this output pixel via this kernel position
            int in_y = out_y_global - kh + pad_h;
            int in_x = out_x_global - kw + pad_w;
            
            // Check bounds for input
            if (in_y >= 0 && in_y < H_in && in_x >= 0 && in_x < W_in) {
                // Iterate over input channels within the group
                for (int c_in_offset = 0; c_in_offset < C_in / groups; ++c_in_offset) {
                    int c_in = c_in_base + c_in_offset;
                    
                    // Load weight: [C_out, C_in/groups, KH, KW]
                    // Index: c_out * (C_in/groups * KH * KW) + c_in_offset * (KH * KW) + kh * KW + kw
                    int w_idx = c_out * (C_in / groups * KH * KW) + 
                                c_in_offset * (KH * KW) + 
                                kh * KW + kw;
                    
                    // Load input: [N, C_in, H_in, W_in]
                    // Index: n * (C_in * H_in * W_in) + c_in * (H_in * W_in) + in_y * W_in + in_x
                    int i_idx = n * (C_in * H_in * W_in) + 
                                c_in * (H_in * W_in) + 
                                in_y * W_in + in_x;
                    
                    sum += weight[w_idx] * input[i_idx];
                }
            }
        }
    }
    
    // Add bias if present
    if (bias != nullptr) {
        sum += bias[c_out];
    }
    
    // Write to output: [N, C_out, H_out, W_out]
    int o_idx = n * (C_out * H_out * W_out) + 
                c_out * (H_out * W_out) + 
                out_y_global * W_out + out_x_global;
    
    output[o_idx] = sum;
}

torch::Tensor transposed_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int out_pad_h, int out_pad_w)
{
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    
    auto N = input.size(0);
    auto C_in = input.size(1);
    auto H_in = input.size(2);
    auto W_in = input.size(3);
    
    auto C_out = weight.size(0);
    auto C_in_w = weight.size(1); // Should be C_in / groups
    auto KH = weight.size(2);
    auto KW = weight.size(3);
    
    TORCH_CHECK(C_in % C_in_w == 0, "Input channels must be divisible by weight input channels");
    int groups = C_in / C_in_w;
    
    // Calculate output dimensions
    int H_out = (H_in - 1) * stride_h - 2 * pad_h + KH + out_pad_h;
    int W_out = (W_in - 1) * stride_w - 2 * pad_w + KW + out_pad_w;
    
    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());
    
    if (N == 0 || C_out == 0 || H_out == 0 || W_out == 0) {
        return output;
    }

    // Define block and grid dimensions
    // We want to cover N * C_out * H_out * W_out elements.
    // Let's use a 4D grid: (N, C_out, ceil(H_out/16), ceil(W_out/32))
    // Block: (32, 16) -> 512 threads per block.
    
    const int block_h = 16;
    const int block_w = 32;
    
    dim3 block(block_w, block_h);
    dim3 grid(N, C_out, 
              (H_out + block_h - 1) / block_h, 
              (W_out + block_w - 1) / block_w);

    // Prepare bias pointer
    const float* bias_ptr = nullptr;
    if (bias.numel() > 0) {
        bias_ptr = bias.data_ptr<float>();
    }

    transposed_conv2d_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, KH, KW,
        stride_h, stride_w,
        pad_h, pad_w,
        out_pad_h, out_pad_w,
        groups
    );

    CUDA_CHECK(cudaGetLastError());
    
    return output;
}
"""

transposed_conv2d_cpp_source = (
    "torch::Tensor transposed_conv2d_cuda("
    "torch::Tensor input,"
    "torch::Tensor weight,"
    "torch::Tensor bias,"
    "int stride_h, int stride_w,"
    "int pad_h, int pad_w,"
    "int out_pad_h, int out_pad_w"
    ");"
);

# Compile the inline CUDA code
transposed_conv2d = load_inline(
    name="transposed_conv2d",
    cpp_sources=transposed_conv2d_cpp_source,
    cuda_sources=transposed_conv2d_source,
    functions=["transposed_conv2d_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Optimized Transposed 2D Convolution using custom CUDA operator.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and bias manually to match PyTorch's initialization if needed,
        # but for the custom op, we just need the tensors.
        # We create a dummy ConvTranspose2d to get the correct weight/bias shapes and initialization
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, output_padding=output_padding, 
            groups=groups, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution using custom CUDA operator.
        """
        weight = self.conv_transpose2d.weight
        bias = self.conv_transpose2d.bias
        
        # Extract parameters
        stride_h, stride_w = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        pad_h, pad_w = self.padding if isinstance(self.padding, tuple) else (self.padding, self.padding)
        out_pad_h, out_pad_w = self.output_padding if isinstance(self.output_padding, tuple) else (self.output_padding, self.output_padding)
        
        # Call custom CUDA kernel
        return transposed_conv2d.transposed_conv2d_cuda(
            x, 
            weight, 
            bias, 
            stride_h, stride_w,
            pad_h, pad_w,
            out_pad_h, out_pad_w
        )