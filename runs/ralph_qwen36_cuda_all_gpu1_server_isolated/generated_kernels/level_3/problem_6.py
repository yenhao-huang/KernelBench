import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernels for Conv2d operations
# We will implement a fused kernel that handles the specific Inception block structure
# to minimize memory transfers and leverage shared memory where possible.
# However, for generalizability and correctness with arbitrary weights, 
# we will implement optimized Conv2d forward passes using cuDNN or custom kernels.
# Given the constraint of "inline embedding", we will write a custom kernel for 
# the specific convolutions involved in this Inception module to demonstrate optimization.
# Note: Writing a fully generic, high-performance Conv2d from scratch in inline CUDA is complex.
# Here, we provide optimized kernels for the 1x1, 3x3, and 5x5 convolutions 
# assuming NHWC or NCHW layout (PyTorch uses NCHW).

custom_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Kernel for 1x1 Convolution (Pointwise)
__global__ void conv1x1_kernel(const float* input, const float* weight, const float* bias, 
                               float* output, int batch_size, int in_channels, int out_channels, 
                               int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * height * width;
    
    if (idx < total_elements) {
        // Calculate indices for output tensor [N, C_out, H, W]
        int w = idx % width;
        int h = (idx / width) % height;
        int c_out = (idx / (width * height)) % out_channels;
        int n = idx / (width * height * out_channels);
        
        float sum = 0.0f;
        if (bias != nullptr) {
            sum = bias[c_out];
        }
        
        // Iterate over input channels
        for (int c_in = 0; c_in < in_channels; ++c_in) {
            // Input index: [N, C_in, H, W]
            int input_idx = n * (in_channels * height * width) + c_in * (height * width) + h * width + w;
            // Weight index: [C_out, C_in, 1, 1] -> flattened as [C_out, C_in]
            int weight_idx = c_out * in_channels + c_in;
            
            sum += input[input_idx] * weight[weight_idx];
        }
        
        output[idx] = sum;
    }
}

// Kernel for 3x3 Convolution (with padding=1)
__global__ void conv3x3_kernel(const float* input, const float* weight, const float* bias, 
                               float* output, int batch_size, int in_channels, int out_channels, 
                               int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * height * width;
    
    if (idx < total_elements) {
        int w = idx % width;
        int h = (idx / width) % height;
        int c_out = (idx / (width * height)) % out_channels;
        int n = idx / (width * height * out_channels);
        
        float sum = 0.0f;
        if (bias != nullptr) {
            sum = bias[c_out];
        }
        
        // Iterate over input channels and 3x3 kernel
        for (int c_in = 0; c_in < in_channels; ++c_in) {
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int ih = h + ky;
                    int iw = w + kx;
                    
                    // Handle padding implicitly by checking bounds (assuming input is padded or we handle it here)
                    // For simplicity in this inline example, we assume the input tensor passed 
                    // already has padding applied or we clamp indices. 
                    // A more robust solution would pad the input before calling this kernel.
                    if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                        int input_idx = n * (in_channels * height * width) + c_in * (height * width) + ih * width + iw;
                        // Weight index: [C_out, C_in, 3, 3] -> flattened as [C_out, C_in, 9]
                        int weight_idx = c_out * in_channels * 9 + c_in * 9 + (ky + 1) * 3 + (kx + 1);
                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        output[idx] = sum;
    }
}

// Kernel for 5x5 Convolution (with padding=2)
__global__ void conv5x5_kernel(const float* input, const float* weight, const float* bias, 
                               float* output, int batch_size, int in_channels, int out_channels, 
                               int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * height * width;
    
    if (idx < total_elements) {
        int w = idx % width;
        int h = (idx / width) % height;
        int c_out = (idx / (width * height)) % out_channels;
        int n = idx / (width * height * out_channels);
        
        float sum = 0.0f;
        if (bias != nullptr) {
            sum = bias[c_out];
        }
        
        // Iterate over input channels and 5x5 kernel
        for (int c_in = 0; c_in < in_channels; ++c_in) {
            for (int ky = -2; ky <= 2; ++ky) {
                for (int kx = -2; kx <= 2; ++kx) {
                    int ih = h + ky;
                    int iw = w + kx;
                    
                    if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                        int input_idx = n * (in_channels * height * width) + c_in * (height * width) + ih * width + iw;
                        // Weight index: [C_out, C_in, 5, 5] -> flattened as [C_out, C_in, 25]
                        int weight_idx = c_out * in_channels * 25 + c_in * 25 + (ky + 2) * 5 + (kx + 2);
                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        output[idx] = sum;
    }
}

// Wrapper functions for PyTorch extension

torch::Tensor conv1x1_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_channels, height, width}, input.options());
    
    const int block_size = 256;
    int total_elements = batch_size * out_channels * height * width;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    float* bias_ptr = bias.numel() > 0 ? bias.data_ptr<float>() : nullptr;
    
    conv1x1_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        bias_ptr, 
        output.data_ptr<float>(), 
        batch_size, in_channels, out_channels, height, width
    );
    
    CUDA_CHECK(cudaGetLastError());
    return output;
}

torch::Tensor conv3x3_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_channels, height, width}, input.options());
    
    const int block_size = 256;
    int total_elements = batch_size * out_channels * height * width;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    float* bias_ptr = bias.numel() > 0 ? bias.data_ptr<float>() : nullptr;
    
    conv3x3_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        bias_ptr, 
        output.data_ptr<float>(), 
        batch_size, in_channels, out_channels, height, width
    );
    
    CUDA_CHECK(cudaGetLastError());
    return output;
}

torch::Tensor conv5x5_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_channels, height, width}, input.options());
    
    const int block_size = 256;
    int total_elements = batch_size * out_channels * height * width;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    float* bias_ptr = bias.numel() > 0 ? bias.data_ptr<float>() : nullptr;
    
    conv5x5_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        bias_ptr, 
        output.data_ptr<float>(), 
        batch_size, in_channels, out_channels, height, width
    );
    
    CUDA_CHECK(cudaGetLastError());
    return output;
}

"""

custom_conv_cpp_source = (
    "torch::Tensor conv1x1_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"
    "torch::Tensor conv3x3_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"
    "torch::Tensor conv5x5_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"
);

# Compile the inline CUDA code
custom_conv_ops = load_inline(
    name="custom_conv_ops",
    cpp_sources=custom_conv_cpp_source,
    cuda_sources=custom_conv_source,
    functions=["conv1x1_cuda", "conv3x3_cuda", "conv5x5_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"]
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        """
        :param in_channels: Number of input channels
        :param out_1x1: Number of output channels for the 1x1 convolution
        :param reduce_3x3: Number of output channels for the 1x1 reduction before 3x3 convolution
        :param out_3x3: Number of output channels for the 3x3 convolution
        :param reduce_5x5: Number of output channels for the 1x1 reduction before 5x5 convolution
        :param out_5x5: Number of output channels for the 5x5 convolution
        :param pool_proj: Number of output channels for the pooling projection
        """
        super(ModelNew, self).__init__()
        
        # Initialize weights and biases manually to match PyTorch Conv2d structure
        # Conv2d weight shape: [out_channels, in_channels, kernel_h, kernel_w]
        # Conv2d bias shape: [out_channels]
        
        # 1x1 convolution branch
        self.weight_1x1 = nn.Parameter(torch.randn(out_1x1, in_channels, 1, 1))
        self.bias_1x1 = nn.Parameter(torch.zeros(out_1x1))
        
        # 3x3 convolution branch (two convs)
        self.weight_3x3_reduce = nn.Parameter(torch.randn(reduce_3x3, in_channels, 1, 1))
        self.bias_3x3_reduce = nn.Parameter(torch.zeros(reduce_3x3))
        
        self.weight_3x3 = nn.Parameter(torch.randn(out_3x3, reduce_3x3, 3, 3))
        self.bias_3x3 = nn.Parameter(torch.zeros(out_3x3))
        
        # 5x5 convolution branch (two convs)
        self.weight_5x5_reduce = nn.Parameter(torch.randn(reduce_5x5, in_channels, 1, 1))
        self.bias_5x5_reduce = nn.Parameter(torch.zeros(reduce_5x5))
        
        self.weight_5x5 = nn.Parameter(torch.randn(out_5x5, reduce_5x5, 5, 5))
        self.bias_5x5 = nn.Parameter(torch.zeros(out_5x5))
        
        # Max pooling branch
        # Pooling is kept as standard PyTorch op for simplicity and correctness
        self.pool_proj_weight = nn.Parameter(torch.randn(pool_proj, in_channels, 1, 1))
        self.pool_proj_bias = nn.Parameter(torch.zeros(pool_proj))

    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        
        # 1x1 branch
        branch1x1 = custom_conv_ops.conv1x1_cuda(x, self.weight_1x1, self.bias_1x1)
        
        # 3x3 branch: 1x1 reduce then 3x3 conv
        # First, we need to handle the 1x1 reduction. We can use our custom kernel or standard torch.
        # To keep it simple and robust, let's use standard torch for intermediate steps if needed,
        # but here we will use custom kernels for all convs.
        
        # Reduce 3x3 branch: Conv2d(in_channels, reduce_3x3, kernel_size=1)
        branch3x3_reduce = custom_conv_ops.conv1x1_cuda(x, self.weight_3x3_reduce, self.bias_3x3_reduce)
        
        # Then Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)
        branch3x3 = custom_conv_ops.conv3x3_cuda(branch3x3_reduce, self.weight_3x3, self.bias_3x3)
        
        # 5x5 branch: 1x1 reduce then 5x5 conv
        # Reduce 5x5 branch: Conv2d(in_channels, reduce_5x5, kernel_size=1)
        branch5x5_reduce = custom_conv_ops.conv1x1_cuda(x, self.weight_5x5_reduce, self.bias_5x5_reduce)
        
        # Then Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)
        branch5x5 = custom_conv_ops.conv5x5_cuda(branch5x5_reduce, self.weight_5x5, self.bias_5x5)
        
        # Max pooling branch
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = custom_conv_ops.conv1x1_cuda(branch_pool, self.pool_proj_weight, self.pool_proj_bias)
        
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)

# Test code setup (not included in output as per instructions, but for reference)
# in_channels = 480
# out_1x1 = 192
# reduce_3x3 = 96
# out_3x3 = 208
# reduce_5x5 = 16
# out_5x5 = 48
# pool_proj = 64
# batch_size = 10
# height = 224
# width = 224

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj]