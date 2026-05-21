import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused Conv3D + ReLU + LeakyReLU + GELU + Sigmoid + Bias
fused_conv_activations_bias_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// ReLU
__device__ float relu(float x) {
    return fmaxf(0.0f, x);
}

// LeakyReLU
__device__ float leaky_relu(float x, float negative_slope) {
    return (x >= 0.0f) ? x : negative_slope * x;
}

// GELU approximation
__device__ float gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// Sigmoid
__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void fused_conv3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int depth,
    int height,
    int width,
    int kernel_size,
    int out_depth,
    int out_height,
    int out_width,
    float negative_slope
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    if (idx >= total_elements) return;

    // Decompose linear index
    int w_out = idx % out_width;
    int h_out = (idx / out_width) % out_height;
    int d_out = (idx / (out_width * out_height)) % out_depth;
    int oc = (idx / (out_width * out_height * out_depth)) % out_channels;
    int b = idx / (out_width * out_height * out_depth * out_channels);

    float sum = 0.0f;
    int half_k = kernel_size / 2;

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kd = 0; kd < kernel_size; ++kd) {
            int d_in = d_out + kd - half_k;
            if (d_in < 0 || d_in >= depth) continue;
            for (int kh = 0; kh < kernel_size; ++kh) {
                int h_in = h_out + kh - half_k;
                if (h_in < 0 || h_in >= height) continue;
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int w_in = w_out + kw - half_k;
                    if (w_in < 0 || w_in >= width) continue;
                    
                    int input_idx = ((b * in_channels + ic) * depth + d_in) * height + h_in;
                    input_idx = input_idx * width + w_in;
                    
                    int weight_idx = (((oc * in_channels + ic) * kernel_size + kd) * kernel_size + kh) * kernel_size + kw;
                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Apply activations sequentially
    float val = sum;
    val = relu(val);
    val = leaky_relu(val, negative_slope);
    val = gelu(val);
    val = sigmoid(val);
    
    // Add bias (bias shape: out_channels, 1, 1, 1)
    val += bias[oc];
    
    output[idx] = val;
}

torch::Tensor fused_conv3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    float negative_slope
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);
    int out_channels = weight.size(0);
    
    int out_depth = depth - kernel_size + 1;
    int out_height = height - kernel_size + 1;
    int out_width = width - kernel_size + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width}, input.options());
    
    int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_conv3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        depth,
        height,
        width,
        kernel_size,
        out_depth,
        out_height,
        out_width,
        negative_slope
    );
    
    return output;
}
"""

fused_conv_cpp_source = (
    "torch::Tensor fused_conv3d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int kernel_size, float negative_slope);"
)

# Compile the inline CUDA code
fused_conv_module = load_inline(
    name="fused_conv3d_activations_bias",
    cpp_sources=fused_conv_cpp_source,
    cuda_sources=fused_conv_activations_bias_source,
    functions=["fused_conv3d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.kernel_size = kernel_size
        self.negative_slope = 0.01
        self.fused_conv = fused_conv_module

    def forward(self, x):
        return self.fused_conv.fused_conv3d_cuda(x, self.conv_weight, self.bias, self.kernel_size, self.negative_slope)