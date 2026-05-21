import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA implementation for 2D Convolution (im2col + GEMM approach)
# This is often faster than cuDNN for small kernels on specific hardware configurations
# or when we want to avoid the overhead of cuDNN dispatching for simple operations.
# However, for large inputs, cuDNN is usually optimal. 
# Given the prompt asks for optimization via custom CUDA, we implement a highly optimized 
# im2col + matrix multiplication approach which can be fused and tuned.

conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Helper to calculate grid/block dimensions
dim3 get_grid_dims(int n) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    return dim3(grid_size);
}

__global__ void im2col_kernel(const float* input, float* col, 
                              const int height, const int width, 
                              const int channels, 
                              const int kernel_h, const int kernel_w,
                              const int pad_h, const int pad_w,
                              const int stride_h, const int stride_w,
                              const int dilation_h, const int dilation_w) {
    // Each thread handles one element of the output column matrix
    // The column matrix has shape (kernel_h * kernel_w * channels, out_h * out_w)
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = kernel_h * kernel_w * channels * height * width / stride_h / stride_w; 
    // Note: This calculation assumes square strides/padding for simplicity in indexing logic below
    
    if (index >= total_elements) return;

    // Decompose index into spatial and channel components
    // Output spatial dimensions
    int out_h = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_w = (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    // Map linear index to (c, kh, kw, oh, ow)
    // col layout: [out_h * out_w] blocks of [kernel_h * kernel_w * channels]
    
    int ow = index % out_w;
    int temp = index / out_w;
    int oh = temp % out_h;
    int c = temp / out_h;

    // Calculate the corresponding input coordinates for each kernel element
    // We need to fill the column vector for this specific (c, oh, ow) position.
    // The column vector stores values from all kernel positions for channel c.
    
    float* col_ptr = col + index * 0; // Base pointer for this output pixel's column
    
    // Actually, standard im2col layout is:
    // col[c * kernel_h * kernel_w * out_h * out_w + ...]
    // Let's use a simpler mapping:
    // Output index in col matrix: (c * kernel_h * kernel_w) * (oh * out_w + ow) + (kh * kernel_w + kw)
    
    int base_col_idx = c * kernel_h * kernel_w * out_h * out_w + oh * out_w + ow;
    
    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int in_h = oh * stride_h - pad_h + kh * dilation_h;
            int in_w = ow * stride_w - pad_w + kw * dilation_w;
            
            // Check bounds
            if (in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
                int input_idx = c * height * width + in_h * width + in_w;
                int col_offset = kh * kernel_w + kw;
                // The position in the column vector for this specific (kh, kw) relative to the start of the pixel's block
                // But wait, standard im2col puts all channels first? Or all spatial?
                // Common layout: [out_h * out_w] x [kernel_h * kernel_w * channels]
                // So index = (oh * out_w + ow) * (kernel_h * kernel_w * channels) + (kh * kernel_w + kw) * channels + c
                
                int col_idx = base_col_idx * (kernel_h * kernel_w * channels) / (out_h * out_w); 
                // Let's recalculate cleanly.
                
                // Layout: [out_h, out_w, kh, kw, c] flattened? No, usually [kh, kw, c] per pixel.
                // Let's stick to: col[out_h*out_w][kernel_h*kernel_w*channels]
                
                int pixel_idx = oh * out_w + ow;
                int kernel_channel_idx = kh * kernel_w * channels + kw * channels + c;
                int final_col_idx = pixel_idx * (kernel_h * kernel_w * channels) + kernel_channel_idx;
                
                col[final_col_idx] = input[input_idx];
            } else {
                // Padding is zero
                int pixel_idx = oh * out_w + ow;
                int kernel_channel_idx = kh * kernel_w * channels + kw * channels + c;
                int final_col_idx = pixel_idx * (kernel_h * kernel_w * channels) + kernel_channel_idx;
                col[final_col_idx] = 0.0f;
            }
        }
    }
}

// Simple GEMM for small matrices or general case
__global__ void gemm_kernel(const float* A, const float* B, float* C, 
                            int M, int N, int K) {
    // A: [M, K], B: [K, N], C: [M, N]
    // Each thread computes one element of C
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight, 
                          int stride_h, int stride_w, int pad_h, int pad_w, 
                          int dilation_h, int dilation_w) {
    // Input: [N, C_in, H, W]
    // Weight: [C_out, C_in, K_h, K_w]
    
    auto N = input.size(0);
    auto C_in = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);
    
    auto C_out = weight.size(0);
    auto K_h = weight.size(2);
    auto K_w = weight.size(3);
    
    auto out_h = (H + 2 * pad_h - dilation_h * (K_h - 1) - 1) / stride_h + 1;
    auto out_w = (W + 2 * pad_w - dilation_w * (K_w - 1) - 1) / stride_w + 1;
    
    // Output tensor [N, C_out, out_h, out_w]
    auto output = torch::zeros({N, C_out, out_h, out_w}, input.options());
    
    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    
    // We will process each batch item separately to keep memory management simple
    for (int n = 0; n < N; ++n) {
        // 1. Im2Col: [C_in, H, W] -> [K_h * K_w * C_in, out_h * out_w]
        int col_size = K_h * K_w * C_in * out_h * out_w;
        auto col_tensor = torch::zeros({col_size}, input.options());
        float* col_ptr = col_tensor.data_ptr<float>();
        
        const int block_size = 256;
        dim3 grid(get_grid_dims(col_size));
        
        im2col_kernel<<<grid, block_size>>>(
            input_ptr + n * C_in * H * W, 
            col_ptr,
            H, W, C_in,
            K_h, K_w,
            pad_h, pad_w,
            stride_h, stride_w,
            dilation_h, dilation_w
        );
        
        // 2. GEMM: [C_out, C_in * K_h * K_w] x [C_in * K_h * K_w, out_h * out_w] -> [C_out, out_h * out_w]
        // Weight is [C_out, C_in * K_h * K_w] if flattened appropriately?
        // PyTorch weight is [C_out, C_in, K_h, K_w]. We need to flatten last 3 dims.
        // Let's assume we pass the raw weight tensor and handle indexing in GEMM or pre-flatten.
        // For simplicity in this inline example, we'll treat weight as [C_out, K_h*K_w*C_in]
        
        int M = C_out;
        int N_gem = out_h * out_w;
        int K_gem = C_in * K_h * K_w;
        
        auto gemm_output = torch::zeros({M, N_gem}, input.options());
        float* gemm_out_ptr = gemm_output.data_ptr<float>();
        
        // Launch GEMM
        dim3 block_gemm(16, 16);
        dim3 grid_gemm((N_gem + 15) / 16, (M + 15) / 16);
        
        gemm_kernel<<<grid_gemm, block_gemm>>>(
            weight_ptr, 
            col_ptr, 
            gemm_out_ptr, 
            M, N_gem, K_gem
        );
        
        // 3. Reshape/Permute: [C_out, out_h * out_w] -> [C_out, out_h, out_w] -> add to output[n]
        // Since we are writing directly to output, we can just copy with stride logic or reshape.
        // torch::reshape is efficient if contiguous.
        auto gemm_reshaped = gemm_output.reshape({C_out, out_h, out_w});
        
        // Copy to output batch slice
        // output[n] = gemm_reshaped
        // We can use copy_ or memcpy
        cudaMemcpyAsync(
            output_ptr + n * C_out * out_h * out_w,
            gemm_out_ptr,
            sizeof(float) * C_out * out_h * out_w,
            cudaMemcpyDeviceToDevice
        );
    }
    
    cudaDeviceSynchronize();
    return output;
}
"""

conv2d_cpp_source = (
    "torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight, "
    "int stride_h, int stride_w, int pad_h, int pad_w, "
    "int dilation_h, int dilation_w);"
);

// Compile the custom CUDA operator
conv2d_module = load_inline(
    name="conv2d_cuda_module",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_source,
    functions=["conv2d_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"]
)

class ModelNew(nn.Module):
    """
    Optimized 2D Convolution using custom CUDA im2col + GEMM.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # We still need the weight tensor for the forward pass.
        # In a real scenario, you might want to initialize this properly.
        # Here we just store the parameters as buffers or module attributes.
        # Since nn.Conv2d handles initialization, we'll mimic its structure but use custom op.
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        # Initialize weight and bias as learnable parameters
        # Using Kaiming uniform initialization similar to PyTorch's default for Conv2d
        fan_in = in_channels * kernel_size * kernel_size
        bound = 1 / (fan_in ** 0.5)
        
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.uniform_(self.weight, -bound, bound)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using custom CUDA operator.
        """
        stride_h = self.stride if isinstance(self.stride, int) else self.stride[0]
        stride_w = self.stride if isinstance(self.stride, int) else self.stride[1]
        pad_h = self.padding if isinstance(self.padding, int) else self.padding[0]
        pad_w = self.padding if isinstance(self.padding, int) else self.padding[1]
        dil_h = self.dilation if isinstance(self.dilation, int) else self.dilation[0]
        dil_w = self.dilation if isinstance(self.dilation, int) else self.dilation[1]
        
        # Call custom CUDA kernel
        out = conv2d_module.conv2d_cuda(x, self.weight, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w)
        
        if self.bias is not None:
            # Add bias: [1, C_out, 1, 1] broadcastable
            out = out + self.bias.view(1, -1, 1, 1)
            
        return out

def get_inputs():
    batch_size = 8
    height = 512
    width = 1024
    in_channels = 64
    out_channels = 128
    
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    in_channels = 64
    out_channels = 128
    kernel_size = 3
    return [in_channels, out_channels, kernel_size]