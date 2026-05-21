import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA implementation for 2D Convolution (im2col + gemm approach)
# This is a simplified but functional implementation focusing on correctness and structure.
# For production, cuDNN or CUTLASS would be used, but here we implement a raw kernel to satisfy the prompt's requirement for custom operators.

conv2d_source = """
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

// Kernel for im2col transformation
__global__ void im2col_kernel(
    const float* input,
    float* col,
    int batch_size,
    int in_channels,
    int height,
    int width,
    int kernel_h,
    int kernel_w,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w) {
    
    // Each thread handles one element of the output column matrix
    // The column matrix has shape: (in_channels * kernel_h * kernel_w, out_h * out_w)
    // We iterate over batch, output spatial positions, and kernel elements
    
    int total_out_elements = height * width; // Assuming stride=1, pad=0 for simplicity in this basic impl, 
                                             // but we need to calculate actual out dimensions.
    
    // Calculate output dimensions
    int out_h = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_w = (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * in_channels * kernel_h * kernel_w * out_h * out_w;
    
    if (idx >= total_elements) return;
    
    // Decompose index into components
    int temp_idx = idx;
    
    // Batch index
    int b = temp_idx / (in_channels * kernel_h * kernel_w * out_h * out_w);
    temp_idx %= (in_channels * kernel_h * kernel_w * out_h * out_w);
    
    // Output spatial index (out_h * out_w)
    int ow = temp_idx / (in_channels * kernel_h * kernel_w * out_h);
    temp_idx %= (in_channels * kernel_h * kernel_w * out_h);
    int oh = temp_idx / (in_channels * kernel_h * kernel_w);
    
    // Kernel element index
    int k_idx = temp_idx / (in_channels * out_h * out_w); // This logic is slightly flawed for general case, let's restart decomposition
    
    // Better decomposition:
    // Index in col matrix: [b, c, kh, kw, oh, ow]
    // Linear index = b * (C*K*H*W) + c * (K*H*W) + kh * (H*W) + kw * (H*W) ... wait, standard im2col layout is usually:
    // col[c * kernel_h * kernel_w * out_h * out_w + kh * kernel_w * out_h * out_w + kw * out_h * out_w + oh * out_w + ow]
    
    // Let's use a simpler thread mapping: One thread per output pixel per channel per kernel element? No, too many threads.
    // Standard approach: One thread per element in the 'col' matrix.
    // col shape: (C * K_h * K_w, H_out * W_out) for single batch. Total: B * C * K_h * K_w * H_out * W_out
    
    int b_idx = idx / (in_channels * kernel_h * kernel_w * out_h * out_w);
    int rem = idx % (in_channels * kernel_h * kernel_w * out_h * out_w);
    
    int c_idx = rem / (kernel_h * kernel_w * out_h * out_w);
    rem %= (kernel_h * kernel_w * out_h * out_w);
    
    int kh_idx = rem / (kernel_w * out_h * out_w);
    rem %= (kernel_w * out_h * out_w);
    
    int kw_idx = rem / (out_h * out_w);
    rem %= (out_h * out_w);
    
    int oh_idx = rem / out_w;
    int ow_idx = rem % out_w;
    
    // Calculate source coordinates in input image
    int ih = oh_idx * stride_h - pad_h + kh_idx * dilation_h;
    int iw = ow_idx * stride_w - pad_w + kw_idx * dilation_w;
    
    float val = 0.0f;
    if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
        // Input layout: N, C, H, W
        int input_idx = b_idx * in_channels * height * width + c_idx * height * width + ih * width + iw;
        val = input[input_idx];
    }
    
    // Output layout for col: [b, c, kh, kw, oh, ow] flattened
    // We need to map this back to the linear index if we were doing it differently, 
    // but here we are writing directly to 'col' which is pre-allocated.
    // The caller needs to know the stride. Let's assume col is laid out as:
    // col[b * (C*K*H*W) + c * (K*H*W) + kh * (H*W) + kw * (H*W) ... ] -> This is complex.
    
    // Simpler Col Layout: [b, c, kh, kw, oh, ow]
    int col_idx = b_idx * in_channels * kernel_h * kernel_w * out_h * out_w 
                + c_idx * kernel_h * kernel_w * out_h * out_w 
                + kh_idx * kernel_w * out_h * out_w 
                + kw_idx * out_h * out_w 
                + oh_idx * out_w 
                + ow_idx;
                
    col[col_idx] = val;
}

// Kernel for GEMM: Col (M x K) * Weight (K x N) -> Output (M x N)
// M = batch_size * out_h * out_w
// K = in_channels * kernel_h * kernel_w
// N = out_channels
__global__ void gemm_kernel(
    const float* col,
    const float* weight,
    float* output,
    int m, // batch_size * out_h * out_w
    int k, // in_channels * kernel_h * kernel_w
    int n, // out_channels
    const float* bias) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * n) return;
    
    int i = idx / n; // row index in output matrix (corresponds to a specific pixel in batch)
    int j = idx % n; // col index in output matrix (corresponds to an output channel)
    
    float sum = 0.0f;
    for (int l = 0; l < k; ++l) {
        sum += col[i * k + l] * weight[l * n + j];
    }
    
    if (bias != nullptr) {
        sum += bias[j];
    }
    
    output[idx] = sum;
}

torch::Tensor conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w) {

    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    
    auto out_channels = weight.size(0);
    auto kernel_h = weight.size(2);
    auto kernel_w = weight.size(3);
    
    // Calculate output dimensions
    int out_h = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_w = (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    
    // Allocate column matrix
    // Shape: [batch_size, in_channels, kernel_h, kernel_w, out_h, out_w]
    int col_size = batch_size * in_channels * kernel_h * kernel_w * out_h * out_w;
    auto col = torch::zeros({col_size}, input.options());
    
    // Allocate output matrix
    // Shape: [batch_size, out_channels, out_h, out_w]
    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, input.options());
    
    const int block_size = 256;
    const int num_blocks_col = (col_size + block_size - 1) / block_size;
    
    // Launch im2col kernel
    im2col_kernel<<<num_blocks_col, block_size>>>(
        input.data_ptr<float>(),
        col.data_ptr<float>(),
        batch_size,
        in_channels,
        height,
        width,
        kernel_h,
        kernel_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w
    );
    
    // Launch GEMM kernel
    int m = batch_size * out_h * out_w;
    int k = in_channels * kernel_h * kernel_w;
    int n = out_channels;
    int output_size = m * n;
    
    const int num_blocks_gemm = (output_size + block_size - 1) / block_size;
    
    gemm_kernel<<<num_blocks_gemm, block_size>>>(
        col.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        m, k, n,
        bias.numel() > 0 ? bias.data_ptr<float>() : nullptr
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
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
    "int pad_h,"
    "int pad_w,"
    "int dilation_h,"
    "int dilation_w"
    ");"
)

# Compile the inline CUDA code
conv2d_module = load_inline(
    name="conv2d_cuda_module",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_source,
    functions=["conv2d_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"]
)


class ModelNew(nn.Module):
    """
    Optimized 2D Convolution using custom CUDA operators (im2col + GEMM).
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # We still need to store the parameters for the forward pass, 
        # but we won't use nn.Conv2d's internal logic.
        # Note: The custom kernel expects weight in [out_channels, in_channels/groups, kernel_h, kernel_w] format.
        # nn.Conv2d stores it this way by default if groups=1.
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using custom CUDA kernels.
        """
        # Handle groups by splitting weight and input if necessary. 
        # For simplicity in this basic implementation, we assume groups=1 as per standard conv2d default 
        # and the prompt's example structure. If groups > 1, a more complex kernel or loop is needed.
        # The provided im2col+gemm approach above assumes standard convolution (groups=1).
        
        if self.groups != 1:
            raise NotImplementedError("Custom CUDA operator currently only supports groups=1 for simplicity.")

        return conv2d_module.conv2d_cuda(
            x,
            self.weight,
            self.bias if self.bias is not None else torch.empty(0),
            self.stride,
            self.stride,
            self.padding,
            self.padding,
            self.dilation,
            self.dilation
        )

# Test code (not included in output as per instructions, but kept for reference)
# batch_size = 16
# in_channels = 16
# out_channels = 128
# kernel_size = 3
# width = 1024
# height = 1024

# def get_inputs():
#     x = torch.rand(batch_size, in_channels, height, width)
#     return [x]

# def get_init_inputs():
#     return [in_channels, out_channels, kernel_size]