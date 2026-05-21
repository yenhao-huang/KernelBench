import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Conv2d + GELU + AdaptiveAvgPool2d + Squeeze
# This fuses the operations to reduce memory traffic and kernel launch overhead.
optimized_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Helper for GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
__device__ __forceinline__ float gelu(float x) {
    float cdf = 0.5f * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
    return x * cdf;
}

__global__ void conv_gelu_pool_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
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
    int pad_w
) {
    // Each thread handles one output element (batch, channel) after pooling
    // Since we are doing global average pooling to 1x1, the spatial dimensions become 1.
    // Total outputs = batch_size * out_channels
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels;

    if (idx >= total_elements) return;

    int b = idx / out_channels;
    int c_out = idx % out_channels;

    float sum = 0.0f;
    
    // Perform Convolution manually
    // Output spatial size is assumed to be 1x1 for the pooling step, 
    // but we need to compute the full conv output first or accumulate directly if we know final pool size.
    // To keep it general and correct, let's assume standard valid/padded conv.
    // However, since we are fusing with AdaptiveAvgPool2d(..., 1), we effectively sum over H_out * W_out.
    
    // Let's compute the convolution output for the single spatial location that remains after pooling?
    // No, AdaptiveAvgPool2d(x, 1) averages over all spatial locations of x.
    // So we need to sum the conv+gelu outputs over all H_out and W_out.
    
    // Calculate output spatial dimensions
    int out_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;

    // We need to iterate over all input channels and spatial locations for the convolution
    // And then average them.
    
    float conv_sum = 0.0f;
    
    // Iterate over output spatial positions
    for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
            float val = 0.0f;
            
            // Iterate over input channels
            for (int ic = 0; ic < in_channels; ++ic) {
                // Calculate input coordinates
                int ih_start = oh * stride_h - pad_h;
                int iw_start = ow * stride_w - pad_w;
                
                float sum_ic = 0.0f;
                
                // Iterate over kernel
                for (int kh = 0; kh < kernel_h; ++kh) {
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        int ih = ih_start + kh;
                        int iw = iw_start + kw;
                        
                        // Check bounds
                        if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                            // Input index: N, C, H, W
                            int input_idx = b * in_channels * height * width + ic * height * width + ih * width + iw;
                            // Weight index: O, C, KH, KW
                            int weight_idx = c_out * in_channels * kernel_h * kernel_w + ic * kernel_h * kernel_w + kh * kernel_w + kw;
                            
                            sum_ic += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
                val += sum_ic;
            }
            
            // Add bias
            if (bias != nullptr) {
                val += bias[c_out];
            }
            
            // Apply GELU
            val = gelu(val);
            
            conv_sum += val;
        }
    }
    
    // Global Average Pooling: divide by number of spatial elements
    float avg = conv_sum / (out_h * out_w);
    
    // Store result
    output[idx] = avg;
}

torch::Tensor optimized_conv_gelu_pool(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    
    auto out_channels = weight.size(0);
    auto kernel_h = weight.size(2);
    auto kernel_w = weight.size(3);
    
    // Assume stride 1 and padding 0 for simplicity as per standard nn.Conv2d defaults if not specified, 
    // but we should extract them from the module or assume standard. 
    // Since we are replacing the operator in a specific model instance, we can hardcode strides/pads 
    // or pass them. For this inline example, let's assume stride=1, pad=0 as is common for small kernels 
    // unless specified otherwise. However, to be robust, we'll use standard valid convolution logic 
    // corresponding to nn.Conv2d with default stride=1 and padding=0.
    
    int stride_h = 1;
    int stride_w = 1;
    int pad_h = 0;
    int pad_w = 0;

    auto output = torch::empty({batch_size, out_channels}, input.options());
    
    const int block_size = 256;
    int total_elements = batch_size * out_channels;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    // Launch kernel
    conv_gelu_pool_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.numel() > 0 ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
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
        pad_w
    );
    
    cudaDeviceSynchronize();
    return output;
}
"""

optimized_ops_cpp_source = (
    "torch::Tensor optimized_conv_gelu_pool(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"
)

# Compile the inline CUDA code
optimized_ops = load_inline(
    name="optimized_ops",
    cpp_sources=optimized_ops_cpp_source,
    cuda_sources=optimized_ops_source,
    functions=["optimized_conv_gelu_pool"],
    verbose=False,
    extra_cflags=["-O2"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution, applies GELU, and then performs global average pooling
    using a custom fused CUDA operator.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        # We still need the parameters to pass to the CUDA kernel
        # Note: In a real scenario, you might want to register these as buffers or handle them differently
        # but for this inline example, we store them in the module.
        self.conv_weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.conv_bias = nn.Parameter(torch.zeros(out_channels))
        
        # Store dimensions for the kernel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
        Returns:
            Output tensor of shape (batch_size, out_channels)
        """
        # Call the fused CUDA operator
        output = optimized_ops.optimized_conv_gelu_pool(x, self.conv_weight, self.conv_bias)
        
        # The output is already (batch_size, out_channels) due to the pooling in the kernel
        return output


def get_inputs():
    # randomly generate input tensors based on the model architecture
    a = torch.rand(batch_size, in_channels, height, width)
    return [a]

def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    return [in_channels, out_channels, kernel_size]