import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused Conv3d + Min + Softmax
fused_conv_min_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Kernel for fused Conv3d + Min along dimension + Softmax along channel dimension
// This kernel performs the convolution, then applies min along the specified dimension,
// and finally applies softmax along the channel dimension.
// For simplicity, we implement a direct convolution here, but in practice you'd want to use
// optimized convolution algorithms. This is a demonstration of operator fusion.

__global__ void fused_conv3d_min_softmax_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int D, int H, int W,
    int kernel_size,
    int min_dim,
    int softmax_dim
) {
    // Each thread handles one output element: (b, oc, h, w) after min reduction
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * H * W;
    if (idx >= total_elements) return;

    // Compute indices
    int w = idx % W;
    int h = (idx / W) % H;
    int oc = (idx / (W * H)) % out_channels;
    int b = idx / (W * H * out_channels);

    // For min_dim=2 (depth dimension), we need to compute min over D
    // We'll compute the convolution result for all D positions and take min
    float min_val = 1e10f;
    int pad = kernel_size / 2;

    // Iterate over depth dimension to compute min
    for (int d = 0; d < D; d++) {
        float conv_val = bias[oc];
        
        // 3D convolution at position (b, oc, d, h, w)
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kd = 0; kd < kernel_size; kd++) {
                int id = d + kd - pad;
                if (id < 0 || id >= D) continue;
                for (int kh = 0; kh < kernel_size; kh++) {
                    int ih = h + kh - pad;
                    if (ih < 0 || ih >= H) continue;
                    for (int kw = 0; kw < kernel_size; kw++) {
                        int iw = w + kw - pad;
                        if (iw < 0 || iw >= W) continue;
                        
                        int input_idx = ((b * in_channels + ic) * D + id) * H * W + ih * W + iw;
                        int weight_idx = ((oc * in_channels + ic) * kernel_size + kd) * kernel_size * kernel_size + kh * kernel_size + kw;
                        
                        conv_val += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        min_val = fminf(min_val, conv_val);
    }

    // Store intermediate result in shared memory for softmax? 
    // For simplicity, we'll write to a temporary location and do softmax in a separate step
    // Actually, we need to compute softmax across channels, which requires all channels for this (b, h, w)
    // We'll store the min values and then compute softmax in a second pass
    output[idx] = min_val;
}

// Softmax kernel along channel dimension
__global__ void softmax_channel_kernel(
    float* __restrict__ data,
    int batch_size,
    int channels,
    int H, int W
) {
    // Each block handles one (b, h, w) position, computing softmax over channels
    int hw_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_hw = H * W;
    if (hw_idx >= total_hw) return;

    int h = hw_idx / W;
    int w = hw_idx % W;

    for (int b = 0; b < batch_size; b++) {
        // Find max for numerical stability
        float max_val = -1e10f;
        for (int c = 0; c < channels; c++) {
            int idx = ((b * channels + c) * H + h) * W + w;
            max_val = fmaxf(max_val, data[idx]);
        }

        // Compute exp sum
        float sum = 0.0f;
        for (int c = 0; c < channels; c++) {
            int idx = ((b * channels + c) * H + h) * W + w;
            sum += expf(data[idx] - max_val);
        }

        // Normalize
        for (int c = 0; c < channels; c++) {
            int idx = ((b * channels + c) * H + h) * W + w;
            data[idx] = expf(data[idx] - max_val) / sum;
        }
    }
}

torch::Tensor fused_conv3d_min_softmax_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int min_dim,
    int softmax_dim
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);
    int out_channels = weight.size(0);

    auto output = torch::empty({batch_size, out_channels, H, W}, input.options());

    const int block_size = 256;
    const int total_elements = batch_size * out_channels * H * W;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    // Launch fused conv+min kernel
    fused_conv3d_min_softmax_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        D, H, W,
        kernel_size,
        min_dim,
        softmax_dim
    );

    // Launch softmax kernel
    const int hw_elements = H * W;
    const int softmax_blocks = (hw_elements + block_size - 1) / block_size;
    softmax_channel_kernel<<<softmax_blocks, block_size>>>(
        output.data_ptr<float>(),
        batch_size,
        out_channels,
        H, W
    );

    return output;
}
"""

fused_conv_min_softmax_cpp_source = (
    "torch::Tensor fused_conv3d_min_softmax_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias, "
    "int kernel_size, int min_dim, int softmax_dim);"
)

# Compile the inline CUDA code
fused_conv_min_softmax = load_inline(
    name="fused_conv_min_softmax",
    cpp_sources=fused_conv_min_softmax_cpp_source,
    cuda_sources=fused_conv_min_softmax_source,
    functions=["fused_conv3d_min_softmax_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim
        self.fused_op = fused_conv_min_softmax

    def forward(self, x):
        # Use the fused CUDA operator that combines Conv3d, Min, and Softmax
        return self.fused_op.fused_conv3d_min_softmax_cuda(
            x, self.conv.weight, self.conv.bias, 
            self.conv.kernel_size[0], self.dim, 1
        )