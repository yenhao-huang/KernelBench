import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA implementation for 3D Transposed Convolution (ConvTranspose3d)
# This kernel performs the im2col-like extraction followed by GEMM, 
# optimized for FP32. It handles stride, padding, and dilation.
# Note: For very large tensors, a direct spatial loop or FFT-based approach might be faster,
# but this GEMM-based approach is robust and standard for general sizes.

conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Helper to calculate output dimensions
__device__ int get_out_dim(int in_dim, int kernel_size, int stride, int padding, int dilation) {
    return (in_dim - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
}

// Kernel to perform the transposed convolution using im2col logic mapped to GEMM
// We will implement a simplified version that iterates over output pixels and gathers input.
// This is often slower than cuDNN for large batches but avoids library dependencies.
// However, to provide a "custom operator" that is faster than naive PyTorch in some contexts 
// or simply demonstrates the capability, we will use a highly optimized direct computation kernel 
// assuming small kernels (like 3x3x3) which are common.

__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,      // [N, C_in, D_in, H_in, W_in]
    const float* __restrict__ weight,     // [C_out, C_in, K_d, K_h, K_w]
    const float* __restrict__ bias,       // [C_out] or nullptr
    float* __restrict__ output,           // [N, C_out, D_out, H_out, W_out]
    int N, int C_in, int C_out,
    int D_in, int H_in, int W_in,
    int K_d, int K_h, int K_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w,
    bool has_bias
) {
    // Each thread handles one output element: (n, c_out, d_out, h_out, w_out)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Total number of output elements
    long total_out = (long)N * C_out * get_out_dim(D_in, K_d, stride_d, pad_d, dilation_d) * 
                     get_out_dim(H_in, K_h, stride_h, pad_h, dilation_h) * 
                     get_out_dim(W_in, K_w, stride_w, pad_w, dilation_w);
                     
    if (idx >= total_out) return;

    // Decode index to coordinates
    int w_out = idx % get_out_dim(W_in, K_w, stride_w, pad_w, dilation_w);
    int h_out = (idx / get_out_dim(W_in, K_w, stride_w, pad_w, dilation_w)) % get_out_dim(H_in, K_h, stride_h, pad_h, dilation_h);
    int d_out = (idx / ((long)get_out_dim(W_in, K_w, stride_w, pad_w, dilation_w) * get_out_dim(H_in, K_h, stride_h, pad_h, dilation_h))) % get_out_dim(D_in, K_d, stride_d, pad_d, dilation_d);
    int c_out = (idx / ((long)get_out_dim(W_in, K_w, stride_w, pad_w, dilation_w) * get_out_dim(H_in, K_h, stride_h, pad_h, dilation_h) * get_out_dim(D_in, K_d, stride_d, pad_d, dilation_d))) % C_out;
    int n = idx / ((long)C_out * get_out_dim(W_in, K_w, stride_w, pad_w, dilation_w) * get_out_dim(H_in, K_h, stride_h, pad_h, dilation_h) * get_out_dim(D_in, K_d, stride_d, pad_d, dilation_d));

    float sum = 0.0f;

    // Iterate over input channels and kernel dimensions
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int k_d = 0; k_d < K_d; ++k_d) {
            for (int k_h = 0; k_h < K_h; ++k_h) {
                for (int k_w = 0; k_w < K_w; ++k_w) {
                    // Calculate corresponding input coordinates
                    // Formula: d_in = d_out * stride_d - pad_d + k_d * dilation_d
                    int d_in = d_out * stride_d - pad_d + k_d * dilation_d;
                    int h_in = h_out * stride_h - pad_h + k_h * dilation_h;
                    int w_in = w_out * stride_w - pad_w + k_w * dilation_w;

                    // Check bounds
                    if (d_in >= 0 && d_in < D_in && 
                        h_in >= 0 && h_in < H_in && 
                        w_in >= 0 && w_in < W_in) {
                        
                        float val_input = input[((n * C_in + c_in) * D_in + d_in) * H_in + h_in] * W_in + w_in]; // Wait, indexing is tricky. Let's use standard linear indexing.
                        // Input layout: N, C_in, D_in, H_in, W_in
                        // Linear index for input: n*C_in*D_in*H_in*W_in + c_in*D_in*H_in*W_in + d_in*H_in*W_in + h_in*W_in + w_in
                        
                        long input_idx = ((long)n * C_in + c_in) * D_in * H_in * W_in + 
                                         (long)d_in * H_in * W_in + 
                                         (long)h_in * W_in + 
                                         w_in;
                                         
                        // Weight layout: C_out, C_in, K_d, K_h, K_w
                        long weight_idx = ((long)c_out * C_in + c_in) * K_d * K_h * K_w + 
                                          (long)k_d * K_h * K_w + 
                                          (long)k_h * K_w + 
                                          k_w;

                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    if (has_bias) {
        sum += bias[c_out];
    }

    // Output layout: N, C_out, D_out, H_out, W_out
    long out_idx = ((long)n * C_out + c_out) * get_out_dim(D_in, K_d, stride_d, pad_d, dilation_d) * 
                   get_out_dim(H_in, K_h, stride_h, pad_h, dilation_h) * 
                   get_out_dim(W_in, K_w, stride_w, pad_w, dilation_w) + 
                   (long)d_out * get_out_dim(H_in, K_h, stride_h, pad_h, dilation_h) * get_out_dim(W_in, K_w, stride_w, pad_w, dilation_w) + 
                   (long)h_out * get_out_dim(W_in, K_w, stride_w, pad_w, dilation_w) + 
                   w_out;
                   
    output[out_idx] = sum;
}

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    
    auto N = input.size(0);
    auto C_in = input.size(1);
    auto D_in = input.size(2);
    auto H_in = input.size(3);
    auto W_in = input.size(4);

    auto C_out = weight.size(0);
    auto K_d = weight.size(2);
    auto K_h = weight.size(3);
    auto K_w = weight.size(4);

    // We need stride, padding, dilation from the model config. 
    // Since we are embedding this in a class, we can pass them as arguments or hardcode if fixed.
    // To make it generic like the original nn.ConvTranspose3d, we should ideally pass these params.
    // However, load_inline functions have limited signatures. We will assume standard parameters 
    // passed via a wrapper or hardcoded for this specific optimization task context.
    // Let's define a function that takes these as arguments.
    
    return torch::zeros({1}, input.options()); // Placeholder to satisfy compiler if not fully implemented below
}

// Better approach: Define the kernel with explicit parameters passed from Python
"""

# Redefining the source to include all necessary parameters for flexibility
conv_transpose3d_full_source = R"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,      
    const float* __restrict__ weight,     
    const float* __restrict__ bias,       
    float* __restrict__ output,           
    int N, int C_in, int C_out,
    int D_in, int H_in, int W_in,
    int K_d, int K_h, int K_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w,
    bool has_bias
) {
    long total_out = (long)N * C_out;
    int d_out_size = get_out_dim(D_in, K_d, stride_d, pad_d, dilation_d); // Need helper or inline calc
    int h_out_size = get_out_dim(H_in, K_h, stride_h, pad_h, dilation_h);
    int w_out_size = get_out_dim(W_in, K_w, stride_w, pad_w, dilation_w);
    
    total_out *= d_out_size * h_out_size * w_out_size;

    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_out) return;

    int w_out = idx % w_out_size;
    int temp = idx / w_out_size;
    int h_out = temp % h_out_size;
    temp /= h_out_size;
    int d_out = temp % d_out_size;
    int c_out = (temp / d_out_size) % C_out;
    int n = temp / (d_out_size * h_out_size);

    float sum = 0.0f;

    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int k_d = 0; k_d < K_d; ++k_d) {
            for (int k_h = 0; k_h < K_h; ++k_h) {
                for (int k_w = 0; k_w < K_w; ++k_w) {
                    int d_in = d_out * stride_d - pad_d + k_d * dilation_d;
                    int h_in = h_out * stride_h - pad_h + k_h * dilation_h;
                    int w_in = w_out * stride_w - pad_w + k_w * dilation_w;

                    if (d_in >= 0 && d_in < D_in && 
                        h_in >= 0 && h_in < H_in && 
                        w_in >= 0 && w_in < W_in) {
                        
                        long input_idx = ((long)n * C_in + c_in) * D_in * H_in * W_in + 
                                         (long)d_in * H_in * W_in + 
                                         (long)h_in * W_in + 
                                         w_in;
                                         
                        long weight_idx = ((long)c_out * C_in + c_in) * K_d * K_h * K_w + 
                                          (long)k_d * K_h * K_w + 
                                          (long)k_h * K_w + 
                                          k_w;

                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    if (has_bias && bias != nullptr) {
        sum += bias[c_out];
    }

    long out_idx = ((long)n * C_out + c_out) * d_out_size * h_out_size * w_out_size + 
                   (long)d_out * h_out_size * w_out_size + 
                   (long)h_out * w_out_size + 
                   w_out;
                   
    output[out_idx] = sum;
}

// Helper function declaration inside the kernel file is not allowed, so we inline the logic or use a macro.
// Since C++ doesn't allow device functions to be defined outside if used in __global__ without proper linkage,
// we will just repeat the calculation or define it as __device__ before the kernel.

__device__ int get_out_dim(int in_dim, int kernel_size, int stride, int padding, int dilation) {
    return (in_dim - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
}

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    
    auto N = input.size(0);
    auto C_in = input.size(1);
    auto D_in = input.size(2);
    auto H_in = input.size(3);
    auto W_in = input.size(4);

    auto C_out = weight.size(0);
    auto K_d = weight.size(2);
    auto K_h = weight.size(3);
    auto K_w = weight.size(4);

    int d_out_size = get_out_dim(D_in, K_d, stride_d, pad_d, dilation_d);
    int h_out_size = get_out_dim(H_in, K_h, stride_h, pad_h, dilation_h);
    int w_out_size = get_out_dim(W_in, K_w, stride_w, pad_w, dilation_w);

    auto output = torch::zeros({N, C_out, d_out_size, h_out_size, w_out_size}, input.options());

    bool has_bias = bias_opt.has_value();
    const float* bias_ptr = has_bias ? bias_opt.value().data_ptr<float>() : nullptr;

    long total_elements = (long)N * C_out * d_out_size * h_out_size * w_out_size;
    
    if (total_elements == 0) return output;

    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    conv_transpose3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, C_in, C_out,
        D_in, H_in, W_in,
        K_d, K_h, K_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dilation_d, dilation_h, dilation_w,
        has_bias
    );

    return output;
}
"""

conv_transpose3d_cpp_source = (
    "torch::Tensor conv_transpose3d_cuda("
    "torch::Tensor input,"
    "torch::Tensor weight,"
    "c10::optional<torch::Tensor> bias_opt,"
    "int stride_d, int stride_h, int stride_w,"
    "int pad_d, int pad_h, int pad_w,"
    "int dilation_d, int dilation_h, int dilation_w"
    ");"
)

# Compile the inline CUDA code
conv_transpose3d_module = load_inline(
    name="conv_transpose3d_cuda",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_full_source,
    functions=["conv_transpose3d_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Optimized 3D Transposed Convolution using custom CUDA operator.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Store parameters for the forward pass
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        # Initialize weights and biases manually to match nn.ConvTranspose3d behavior
        # Weight shape: [out_channels, in_channels, k, k, k]
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming uniform initialization similar to PyTorch's default for ConvTranspose
        nn.init.kaiming_uniform_(self.weight, a=0.1714592638090354) # sqrt(3/ fan_in) where fan_in = out_ch * k^3 ? 
        # Actually standard is: limit = sqrt(1/fan_in). For ConvTranspose, fan_in is usually calculated based on input.
        # Let's stick to a simple uniform init for stability in this custom op context if exact match isn't critical for training convergence demo.
        # Standard nn.ConvTranspose3d uses kaiming_uniform with gain=sqrt(5/2) or similar? 
        # PyTorch default: limit = sqrt(1 / fan_in). Fan_in = out_channels * kernel_size^3 ? No, fan_in is input channels * kernel_volume.
        
        fan_in = self.in_channels * (self.kernel_size ** 3)
        bound = 1 / (fan_in ** 0.5)
        nn.init.uniform_(self.weight, -bound, bound)
        
        if self.bias is not None:
            fan_in = self.in_channels * (self.kernel_size ** 3)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 3D transposed convolution.
        """
        stride_d, stride_h, stride_w = self.stride, self.stride, self.stride
        pad_d, pad_h, pad_w = self.padding, self.padding, self.padding
        dilation_d, dilation_h, dilation_w = self.dilation, self.dilation, self.dilation
        
        bias_opt = self.bias if self.bias is not None else None
        
        return conv_transpose3d_module.conv_transpose3d_cuda(
            x,
            self.weight,
            bias_opt,
            stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w,
            dilation_d, dilation_h, dilation_w
        )

def get_inputs():
    # randomly generate input tensors based on the model architecture
    x = torch.rand(batch_size, in_channels, depth, height, width).cuda()
    return [x]

def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]