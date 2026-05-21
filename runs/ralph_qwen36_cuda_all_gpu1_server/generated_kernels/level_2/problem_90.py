import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation:
# 1. Conv3d (im2col + gemm or direct convolution)
# 2. LeakyReLU
# 3. Add bias/sum_tensor
# 4. Clamp
# 5. GELU
#
# Note: Implementing a full optimized Conv3d from scratch in inline CUDA is extremely complex 
# and usually slower than cuDNN unless using specific tiling strategies. 
# However, the prompt asks to replace operators for speedups. 
# A common optimization pattern is fusing the activation and element-wise ops after the conv.
# Since replacing the heavy Conv3d with a naive CUDA kernel is often not faster than cuDNN,
# we will focus on fusing the post-convolution operations (LeakyReLU + Add + Clamp + GELU) 
# into a single kernel to reduce memory bandwidth overhead and launch overhead.
# We will keep the Conv3d as it is highly optimized by PyTorch/cuDNN, but fuse the rest.

fused_post_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Helper for GELU approximation or exact calculation
__device__ inline float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
}

// Helper for LeakyReLU
__device__ inline float leaky_relu(float x, float slope) {
    return x > 0.0f ? x : slope * x;
}

// Helper for Clamp
__device__ inline float clamp_val(float x, float min_val, float max_val) {
    if (x < min_val) return min_val;
    if (x > max_val) return max_val;
    return x;
}

__global__ void fused_post_conv_kernel(
    const float* input, 
    const float* sum_tensor, 
    float* output, 
    int batch_size, 
    int channels, 
    int spatial_size, // depth * height * width
    float leaky_slope,
    float clamp_min,
    float clamp_max
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * spatial_size;

    if (idx < total_elements) {
        // Calculate coordinates
        int remaining = idx;
        int c = remaining % channels;
        remaining /= channels;
        int s = remaining % spatial_size;
        int b = remaining / spatial_size;

        // Base indices
        int input_idx = (b * channels + c) * spatial_size + s;
        
        // Load value
        float val = input[input_idx];

        // 1. LeakyReLU
        val = leaky_relu(val, leaky_slope);

        // 2. Add sum_tensor (broadcasted over batch and spatial dims)
        // sum_tensor shape: (out_channels, 1, 1, 1) -> effectively just channel index c
        float bias_val = sum_tensor[c];
        val += bias_val;

        // 3. Clamp
        val = clamp_val(val, clamp_min, clamp_max);

        // 4. GELU
        val = gelu(val);

        output[idx] = val;
    }
}

torch::Tensor fused_post_conv_cuda(
    torch::Tensor conv_output, 
    torch::Tensor sum_tensor,
    float leaky_slope,
    float clamp_min,
    float clamp_max
) {
    // conv_output shape: (N, C, D, H, W)
    auto batch_size = conv_output.size(0);
    auto channels = conv_output.size(1);
    auto depth = conv_output.size(2);
    auto height = conv_output.size(3);
    auto width = conv_output.size(4);
    
    int spatial_size = depth * height * width;
    int total_elements = batch_size * channels * spatial_size;

    auto output = torch::empty_like(conv_output);

    if (total_elements == 0) {
        return output;
    }

    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_post_conv_kernel<<<num_blocks, block_size>>>(
        conv_output.data_ptr<float>(),
        sum_tensor.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        spatial_size,
        leaky_slope,
        clamp_min,
        clamp_max
    );

    return output;
}
"""

fused_post_conv_cpp_source = (
    "torch::Tensor fused_post_conv_cuda("
    "torch::Tensor conv_output, "
    "torch::Tensor sum_tensor, "
    "float leaky_slope, "
    "float clamp_min, "
    "float clamp_max"
    ");"
)

# Compile the inline CUDA code
fused_post_conv = load_inline(
    name="fused_post_conv",
    cpp_sources=fused_post_conv_cpp_source,
    cuda_sources=fused_post_conv_source,
    functions=["fused_post_conv_cuda"],
    verbose=False,
    extra_cflags=["-O2"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D convolution, then fuses LeakyReLU, Add, Clamp, and GELU 
    into a single custom CUDA kernel to reduce memory traffic and kernel launch overhead.
    """
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))
        
        # Hyperparameters matching the original model
        self.leaky_slope = 0.2
        self.clamp_min = -1.0
        self.clamp_max = 1.0

    def forward(self, x):
        # Step 1: Conv3d (Kept as standard PyTorch op for optimal cuDNN usage)
        x = self.conv(x)
        
        # Step 2-5: Fused LeakyReLU + Add + Clamp + GELU
        x = fused_post_conv.fused_post_conv_cuda(
            x, 
            self.sum_tensor, 
            self.leaky_slope, 
            self.clamp_min, 
            self.clamp_max
        )
        
        return x


def get_inputs():
    # randomly generate input tensors based on the model architecture
    batch_size = 128
    in_channels = 8
    depth, height, width = 16, 64, 64
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]


def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    batch_size = 128
    in_channels = 8
    out_channels = 64
    depth, height, width = 16, 64, 64
    kernel_size = 3
    sum_tensor_shape = (out_channels, 1, 1, 1)
    return [in_channels, out_channels, kernel_size, sum_tensor_shape]