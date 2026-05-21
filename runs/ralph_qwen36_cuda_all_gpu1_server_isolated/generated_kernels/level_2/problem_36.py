import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation:
# 1. ConvTranspose2d (simplified for demonstration, but typically this is the bottleneck)
#    However, writing a full generic ConvTranspose2d in inline CUDA is extremely complex and error-prone due to im2col/gather logic.
#    Instead, we will focus on optimizing the post-processing chain which is often memory-bound or has high kernel launch overhead:
#    min(dim=1), sum(dim=2), gelu, add_bias.
#    
#    Actually, the prompt asks to replace operators to get speedups. 
#    ConvTranspose2d is heavy. But implementing it from scratch in inline CUDA is not feasible for a general solution in this context without external libraries like CUTLASS.
#    Let's look at the remaining operations: min, sum, gelu, add.
#    These can be fused into a single kernel to reduce memory traffic (HBM reads/writes).
#    
#    The input to this fused kernel will be the output of ConvTranspose2d.
#    We assume the user has already optimized or is using cuDNN for the conv part, but we replace the subsequent chain.
#    
#    Fused Kernel Logic:
#    Input: x (B, C, H, W)
#    Output: y (B, 1, 1, W) after min(C), sum(H), gelu, add_bias.
#    
#    Wait, the original code does:
#    x = conv_transpose(x) -> (B, out_channels, new_H, new_W)
#    x = torch.min(x, dim=1, keepdim=True)[0] -> (B, 1, new_H, new_W)
#    x = torch.sum(x, dim=2, keepdim=True) -> (B, 1, 1, new_W)
#    x = gelu(x)
#    x = x + bias
    
#    We can fuse min, sum, gelu, and add into one kernel.

fused_post_process_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
__device__ __forceinline__ float gelu(float x) {
    float cdf = 0.5f * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
    return x * cdf;
}

__global__ void fused_min_sum_gelu_add_kernel(
    const float* __restrict__ input, 
    float* __restrict__ output, 
    const float* __restrict__ bias,
    int batch_size, 
    int channels, 
    int height, 
    int width
) {
    // Each thread handles one element in the final output: (b, 1, 1, w)
    // Total output elements = batch_size * width
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * width;

    if (idx < total_elements) {
        int b = idx / width;
        int w = idx % width;

        // We need to compute min over channels and sum over height for this specific (b, w)
        // Initialize min with a large value
        float min_val = 1e38f;
        float sum_val = 0.0f;

        // Loop over channels and height
        // The input tensor is contiguous: [B, C, H, W]
        // Stride for C is H*W, Stride for H is W
        
        int base_idx = b * channels * height * width + w; // Start of the slice for this batch and width index? 
        // No, let's calculate indices properly.
        // Element at (b, c, h, w) is at: b*C*H*W + c*H*W + h*W + w
        
        int base_b = b * channels * height * width;
        
        for (int c = 0; c < channels; ++c) {
            int c_offset = c * height * width;
            for (int h = 0; h < height; ++h) {
                int h_offset = h * width;
                float val = input[base_b + c_offset + h_offset + w];
                
                if (val < min_val) {
                    min_val = val;
                }
                sum_val += val;
            }
        }
        
        // Note: The original code does min(dim=1) then sum(dim=2).
        // Min is over C. Sum is over H.
        // So we take the minimum value across all channels for each (b, h, w)? 
        // No, torch.min(x, dim=1) returns the minimum along dimension 1 (channels).
        // If x is (B, C, H, W), min(dim=1) results in (B, 1, H, W).
        // The value at (b, 0, h, w) is min_{c} x[b, c, h, w].
        
        // Then sum(dim=2) sums over height.
        // So for a fixed (b, w), we sum the values min_{c} x[b, c, h, w] for all h.
        
        // Let's re-evaluate the loop above.
        // The previous loop calculated:
        // min_val = min_{c,h} input[...] -> This is wrong. It should be min over C for each H, then sum over H.
        
        // Correct Logic:
        // 1. For each h in [0, height):
        //      val_h = min_{c in [0, channels)} input[b, c, h, w]
        // 2. Sum all val_h.
        
        float final_sum = 0.0f;
        
        for (int h = 0; h < height; ++h) {
            int base_h = b * channels * height * width + h * width + w;
            float current_min = 1e38f;
            
            // Find min over channels for this specific h and w
            for (int c = 0; c < channels; ++c) {
                int idx_c = base_h + c * height * width;
                float val = input[idx_c];
                if (val < current_min) {
                    current_min = val;
                }
            }
            final_sum += current_min;
        }
        
        // Apply GELU
        float gelu_val = gelu(final_sum);
        
        // Add Bias
        // Bias shape is (1, 1, 1) in the example, but generally it broadcasts.
        // The code uses self.bias which is a Parameter. In the forward pass: x + self.bias.
        // If bias is (1, 1, 1), it adds to all elements.
        float bias_val = bias[0]; 
        
        output[idx] = gelu_val + bias_val;
    }
}

torch::Tensor fused_post_process_cuda(torch::Tensor x, torch::Tensor bias) {
    // x shape: (B, C, H, W)
    auto batch_size = x.size(0);
    auto channels = x.size(1);
    auto height = x.size(2);
    auto width = x.size(3);

    // Output shape: (B, 1, 1, W) -> flattened size is B * W
    auto out_size = batch_size * width;
    auto output = torch::zeros({batch_size, 1, 1, width}, x.options());

    const int block_size = 256;
    const int num_blocks = (out_size + block_size - 1) / block_size;

    fused_min_sum_gelu_add_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        bias.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width
    );

    return output;
}
"""

fused_post_process_cpp_source = (
    "torch::Tensor fused_post_process_cuda(torch::Tensor x, torch::Tensor bias);"
)

# Compile the inline CUDA code
fused_post_process = load_inline(
    name="fused_post_process",
    cpp_sources=fused_post_process_cpp_source,
    cuda_sources=fused_post_process_source,
    functions=["fused_post_process_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA operator for the post-processing chain.
    The ConvTranspose2d is left to PyTorch/cuDNN as it is highly optimized and complex to reimplement from scratch in inline CUDA.
    The subsequent operations (min, sum, gelu, add) are fused into a single kernel to reduce memory bandwidth usage.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # Convolution Transpose (Standard PyTorch implementation)
        x = self.conv_transpose(x)
        
        # Fused Custom CUDA Operator: min(dim=1), sum(dim=2), gelu, add_bias
        x = fused_post_process.fused_post_process_cuda(x, self.bias)
        
        return x

# Helper functions to match the interface expected by the prompt structure
def get_inputs():
    batch_size = 16
    in_channels = 64
    height, width = 128, 128
    # Output of conv_transpose with kernel=3, stride=2, padding=1, output_padding=1 on 128x128 input:
    # H_out = (H_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    # H_out = (128 - 1) * 2 - 2 * 1 + 1 * (3 - 1) + 1 + 1 = 127*2 - 2 + 2 + 1 + 1 = 254 - 2 + 4 = 256
    # W_out is same.
    out_height, out_width = 256, 256
    out_channels = 128
    
    x = torch.rand(batch_size, in_channels, height, width).cuda()
    return [x]

def get_init_inputs():
    batch_size = 16
    in_channels = 64
    out_channels = 128
    kernel_size = 3
    stride = 2
    padding = 1
    output_padding = 1
    bias_shape = (1, 1, 1)
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]