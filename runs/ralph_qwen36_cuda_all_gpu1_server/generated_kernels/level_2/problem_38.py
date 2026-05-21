import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernels for the optimized operations
custom_cuda_source = """
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

// 1. Custom AvgPool3d Kernel
__global__ void avg_pool_3d_kernel(
    const float* input, 
    float* output, 
    int batch_size, 
    int channels, 
    int depth_in, 
    int height_in, 
    int width_in, 
    int kernel_size, 
    int stride, 
    int padding, 
    int depth_out, 
    int height_out, 
    int width_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * depth_out * height_out * width_out;

    if (idx >= total_elements) return;

    // Calculate indices for output tensor
    int w_idx = idx % width_out;
    int h_idx = (idx / width_out) % height_out;
    int d_idx = (idx / (width_out * height_out)) % depth_out;
    int c_idx = (idx / (width_out * height_out * depth_out)) % channels;
    int b_idx = idx / (width_out * height_out * depth_out * channels);

    // Calculate corresponding input coordinates
    int w_start = w_idx * stride - padding;
    int h_start = h_idx * stride - padding;
    int d_start = d_idx * stride - padding;

    float sum = 0.0f;
    int count = 0;

    for (int k = 0; k < kernel_size; ++k) {
        for (int j = 0; j < kernel_size; ++j) {
            for (int i = 0; i < kernel_size; ++i) {
                int w_in = w_start + i;
                int h_in = h_start + j;
                int d_in = d_start + k;

                if (d_in >= 0 && d_in < depth_in && 
                    h_in >= 0 && h_in < height_in && 
                    w_in >= 0 && w_in < width_in) {
                    
                    int input_idx = ((b_idx * channels + c_idx) * depth_in + d_in) * height_in + h_in;
                    input_idx = input_idx * width_in + w_in;
                    sum += input[input_idx];
                    count++;
                }
            }
        }
    }

    if (count > 0) {
        output[idx] = sum / count;
    } else {
        output[idx] = 0.0f;
    }
}

// 2. Custom ConvTranspose3d Kernel (Simplified for stride=1, padding=0 case logic, 
//    but here we implement a general version handling the specific parameters passed)
// Note: Implementing a fully generic and optimized ConvTranspose3d from scratch is complex.
// We will use a standard approach mapping output pixels to input contributions.

__global__ void conv_transpose_3d_kernel(
    const float* input, 
    const float* weight, 
    const float* bias, // Can be null if no bias
    float* output, 
    int batch_size, 
    int in_channels, 
    int out_channels, 
    int depth_in, 
    int height_in, 
    int width_in, 
    int kernel_depth, 
    int kernel_height, 
    int kernel_width, 
    int stride, 
    int padding, 
    int output_padding,
    int depth_out, 
    int height_out, 
    int width_out
) {
    // Each thread handles one element of the output tensor
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * depth_out * height_out * width_out;

    if (idx >= total_elements) return;

    // Decompose index
    int w_idx = idx % width_out;
    int h_idx = (idx / width_out) % height_out;
    int d_idx = (idx / (width_out * height_out)) % depth_out;
    int c_out = (idx / (width_out * height_out * depth_out)) % out_channels;
    int b_idx = idx / (width_out * height_out * depth_out * out_channels);

    float sum = 0.0f;

    // Iterate over input channels and kernel dimensions
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int k_d = 0; k_d < kernel_depth; ++k_d) {
            for (int k_h = 0; k_h < kernel_height; ++k_h) {
                for (int k_w = 0; k_w < kernel_width; ++k_w) {
                    // Calculate the corresponding input coordinate
                    int w_in = w_idx / stride - padding + k_w;
                    int h_in = h_idx / stride - padding + k_h;
                    int d_in = d_idx / stride - padding + k_d;

                    // Check bounds for input tensor
                    if (d_in >= 0 && d_in < depth_in && 
                        h_in >= 0 && h_in < height_in && 
                        w_in >= 0 && w_in < width_in) {
                        
                        // Weight index: (out_channels, in_channels, k_d, k_h, k_w)
                        int weight_idx = ((c_out * in_channels + c_in) * kernel_depth + k_d) * kernel_height + k_h;
                        weight_idx = weight_idx * kernel_width + k_w;

                        // Input index: (batch, in_channels, depth, height, width)
                        int input_idx = ((b_idx * in_channels + c_in) * depth_in + d_in) * height_in + h_in;
                        input_idx = input_idx * width_in + w_in;

                        sum += weight[weight_idx] * input[input_idx];
                    }
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[c_out];
    }

    output[idx] = sum;
}

// 3. Custom Clamp Kernel
__global__ void clamp_kernel(
    const float* input, 
    float* output, 
    int size, 
    float min_val, 
    float max_val
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        if (val < min_val) val = min_val;
        if (val > max_val) val = max_val;
        output[idx] = val;
    }
}

// 4. Custom Softmax Kernel along dim=2 (flattened spatial dimension)
__global__ void softmax_kernel(
    const float* input, 
    float* output, 
    int batch_size, 
    int channels, 
    int seq_len
) {
    // Each block handles one channel of one batch item
    int idx = blockIdx.x;
    if (idx >= batch_size * channels) return;

    int offset = idx * seq_len;
    
    // Find max for numerical stability
    float max_val = -1e20f;
    for (int i = 0; i < seq_len; ++i) {
        if (input[offset + i] > max_val) {
            max_val = input[offset + i];
        }
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < seq_len; ++i) {
        float exp_val = expf(input[offset + i] - max_val);
        output[offset + i] = exp_val;
        sum += exp_val;
    }

    // Normalize
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < seq_len; ++i) {
        output[offset + i] *= inv_sum;
    }
}

// 5. Custom Scale Kernel (Element-wise multiplication by learnable scale)
__global__ void scale_kernel(
    const float* input, 
    const float* scale, 
    float* output, 
    int batch_size, 
    int channels, 
    int spatial_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * spatial_size;

    if (idx < total_elements) {
        // Calculate channel index for broadcasting scale
        int c_idx = idx / spatial_size;
        output[idx] = input[idx] * scale[c_idx];
    }
}

// Python Interface Functions

torch::Tensor custom_avg_pool_3d(
    torch::Tensor x, 
    int kernel_size, 
    int stride, 
    int padding
) {
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(x.dim() == 5, "Input must be 5D tensor (B, C, D, H, W)");

    auto batch_size = x.size(0);
    auto channels = x.size(1);
    auto depth_in = x.size(2);
    auto height_in = x.size(3);
    auto width_in = x.size(4);

    // Calculate output dimensions for AvgPool3d
    auto depth_out = (depth_in + 2 * padding - kernel_size) / stride + 1;
    auto height_out = (height_in + 2 * padding - kernel_size) / stride + 1;
    auto width_out = (width_in + 2 * padding - kernel_size) / stride + 1;

    auto out = torch::zeros({batch_size, channels, depth_out, height_out, width_out}, x.options());

    const int block_size = 256;
    int total_elements = batch_size * channels * depth_out * height_out * width_out;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    avg_pool_3d_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), 
        out.data_ptr<float>(), 
        batch_size, channels, depth_in, height_in, width_in, 
        kernel_size, stride, padding, 
        depth_out, height_out, width_out
    );

    CUDA_CHECK(cudaGetLastError());
    return out;
}

torch::Tensor custom_conv_transpose_3d(
    torch::Tensor x, 
    torch::Tensor weight, 
    torch::Tensor bias,
    int stride, 
    int padding, 
    int output_padding
) {
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight must be on CUDA");

    auto batch_size = x.size(0);
    auto in_channels = x.size(1);
    auto depth_in = x.size(2);
    auto height_in = x.size(3);
    auto width_in = x.size(4);

    auto out_channels = weight.size(0);
    auto kernel_depth = weight.size(2);
    auto kernel_height = weight.size(3);
    auto kernel_width = weight.size(4);

    // Calculate output dimensions for ConvTranspose3d
    auto depth_out = (depth_in - 1) * stride - 2 * padding + kernel_depth + output_padding;
    auto height_out = (height_in - 1) * stride - 2 * padding + kernel_height + output_padding;
    auto width_out = (width_in - 1) * stride - 2 * padding + kernel_width + output_padding;

    auto out = torch::zeros({batch_size, out_channels, depth_out, height_out, width_out}, x.options());

    const int block_size = 256;
    int total_elements = batch_size * out_channels * depth_out * height_out * width_out;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    float* bias_ptr = bias.numel() > 0 ? bias.data_ptr<float>() : nullptr;

    conv_transpose_3d_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        bias_ptr,
        out.data_ptr<float>(), 
        batch_size, in_channels, out_channels, 
        depth_in, height_in, width_in, 
        kernel_depth, kernel_height, kernel_width, 
        stride, padding, output_padding,
        depth_out, height_out, width_out
    );

    CUDA_CHECK(cudaGetLastError());
    return out;
}

torch::Tensor custom_clamp(torch::Tensor x, float min_val, float max_val) {
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");
    
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    clamp_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), 
        out.data_ptr<float>(), 
        size, 
        min_val, 
        max_val
    );

    CUDA_CHECK(cudaGetLastError());
    return out;
}

torch::Tensor custom_softmax_spatial(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");
    
    auto b = x.size(0);
    auto c = x.size(1);
    auto seq_len = x.size(2); // Flattened spatial dimension

    auto out = torch::empty_like(x);

    int total_elements = b * c;
    const int block_size = 1; // One thread per channel/batch pair for simplicity in this kernel structure
    int num_blocks = (total_elements + block_size - 1) / block_size;

    softmax_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), 
        out.data_ptr<float>(), 
        b, c, seq_len
    );

    CUDA_CHECK(cudaGetLastError());
    return out;
}

torch::Tensor custom_scale(torch::Tensor x, torch::Tensor scale) {
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(scale.is_cuda(), "Scale must be on CUDA");

    auto b = x.size(0);
    auto c = x.size(1);
    auto spatial_size = x.numel() / (b * c);

    auto out = torch::empty_like(x);

    const int block_size = 256;
    int total_elements = b * c * spatial_size;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    scale_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), 
        scale.data_ptr<float>(), 
        out.data_ptr<float>(), 
        b, c, spatial_size
    );

    CUDA_CHECK(cudaGetLastError());
    return out;
}
"""

custom_cpp_source = """
#include <torch/extension.h>

torch::Tensor custom_avg_pool_3d(torch::Tensor x, int kernel_size, int stride, int padding);
torch::Tensor custom_conv_transpose_3d(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int stride, int padding, int output_padding);
torch::Tensor custom_clamp(torch::Tensor x, float min_val, float max_val);
torch::Tensor custom_softmax_spatial(torch::Tensor x);
torch::Tensor custom_scale(torch::Tensor x, torch::Tensor scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_avg_pool_3d", &custom_avg_pool_3d, "Custom AvgPool3d");
    m.def("custom_conv_transpose_3d", &custom_conv_transpose_3d, "Custom ConvTranspose3d");
    m.def("custom_clamp", &custom_clamp, "Custom Clamp");
    m.def("custom_softmax_spatial", &custom_softmax_spatial, "Custom Softmax Spatial");
    m.def("custom_scale", &custom_scale, "Custom Scale");
}
"""

# Load the custom extension
model_custom_ops = load_inline(
    name="model_custom_ops",
    cpp_sources=custom_cpp_source,
    cuda_sources=custom_cuda_source,
    functions=["custom_avg_pool_3d", "custom_conv_transpose_3d", "custom_clamp", "custom_softmax_spatial", "custom_scale"],
    verbose=True,
    extra_cflags=["-O2"],
    extra_ldflags=["-lcudart"]
)


class ModelNew(nn.Module):
    """
    Optimized Model using custom CUDA operators.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        
        # Store parameters for the forward pass
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.pool_kernel_size = pool_kernel_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        
        # Initialize learnable parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth, height, width).
        """
        # 1. Custom AvgPool3d
        x = model_custom_ops.custom_avg_pool_3d(x, self.pool_kernel_size, self.stride, self.padding)
        
        # 2. Custom ConvTranspose3d
        x = model_custom_ops.custom_conv_transpose_3d(
            x, 
            self.weight, 
            self.bias, 
            self.stride, 
            self.padding, 
            self.output_padding
        )
        
        # 3. Custom Clamp
        x = model_custom_ops.custom_clamp(x, self.clamp_min, self.clamp_max)
        
        # 4. Flatten spatial dims and apply Softmax
        b, c, d, h, w = x.shape
        x = x.view(b, c, -1)                     # flatten spatial dims: (B, C, D*H*W)
        x = model_custom_ops.custom_softmax_spatial(x)
        
        # 5. Reshape back and apply Scale
        x = x.view(b, c, d, h, w)
        x = model_custom_ops.custom_scale(x, self.scale)
        
        return x