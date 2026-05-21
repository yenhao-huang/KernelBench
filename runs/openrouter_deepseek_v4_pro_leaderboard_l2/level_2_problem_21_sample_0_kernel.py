import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused conv + bias + scale + sigmoid + group norm
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_bias_scale_sigmoid_groupnorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ scale,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int num_groups,
    int H_out,
    int W_out
) {
    // Each thread handles one output element (batch, oc, h, w)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * H_out * W_out;
    if (idx >= total_elements) return;

    // Compute indices
    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int oc = (idx / (W_out * H_out)) % out_channels;
    int n = idx / (W_out * H_out * out_channels);

    // Convolution
    float sum = 0.0f;
    int pad = kernel_size / 2;
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int h_in = h_out + ky - pad;
                int w_in = w_out + kx - pad;
                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    int input_idx = ((n * in_channels + ic) * height + h_in) * width + w_in;
                    int weight_idx = ((oc * in_channels + ic) * kernel_size + ky) * kernel_size + kx;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Add bias
    sum += bias[oc];

    // Multiply by scale
    sum *= scale[oc];

    // Sigmoid
    sum = 1.0f / (1.0f + expf(-sum));

    // Group normalization
    int groups_per_channel = out_channels / num_groups;
    int group_idx = oc / groups_per_channel;
    int group_start = group_idx * groups_per_channel;
    int group_end = group_start + groups_per_channel;

    // Compute mean and variance for the group across spatial dimensions
    float mean = 0.0f;
    float var = 0.0f;
    int spatial_size = H_out * W_out;
    
    // First pass: compute mean
    for (int c = group_start; c < group_end; ++c) {
        for (int h = 0; h < H_out; ++h) {
            for (int w = 0; w < W_out; ++w) {
                int out_idx = ((n * out_channels + c) * H_out + h) * W_out + w;
                // We need the value after sigmoid for all elements in the group
                // Since we're computing this per-element, we need to recompute or store
                // For simplicity, we'll recompute the sigmoid value for each element in the group
                // This is inefficient but demonstrates the fusion concept
                // In practice, you'd use shared memory or multiple passes
                float val;
                if (c == oc && h == h_out && w == w_out) {
                    val = sum;
                } else {
                    // Recompute convolution for other elements in the group
                    float temp_sum = 0.0f;
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int ky = 0; ky < kernel_size; ++ky) {
                            for (int kx = 0; kx < kernel_size; ++kx) {
                                int h_in = h + ky - pad;
                                int w_in = w + kx - pad;
                                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                                    int input_idx = ((n * in_channels + ic) * height + h_in) * width + w_in;
                                    int weight_idx = ((c * in_channels + ic) * kernel_size + ky) * kernel_size + kx;
                                    temp_sum += input[input_idx] * weight[weight_idx];
                                }
                            }
                        }
                    }
                    temp_sum += bias[c];
                    temp_sum *= scale[c];
                    val = 1.0f / (1.0f + expf(-temp_sum));
                }
                mean += val;
            }
        }
    }
    mean /= (groups_per_channel * spatial_size);

    // Second pass: compute variance
    for (int c = group_start; c < group_end; ++c) {
        for (int h = 0; h < H_out; ++h) {
            for (int w = 0; w < W_out; ++w) {
                float val;
                if (c == oc && h == h_out && w == w_out) {
                    val = sum;
                } else {
                    float temp_sum = 0.0f;
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int ky = 0; ky < kernel_size; ++ky) {
                            for (int kx = 0; kx < kernel_size; ++kx) {
                                int h_in = h + ky - pad;
                                int w_in = w + kx - pad;
                                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                                    int input_idx = ((n * in_channels + ic) * height + h_in) * width + w_in;
                                    int weight_idx = ((c * in_channels + ic) * kernel_size + ky) * kernel_size + kx;
                                    temp_sum += input[input_idx] * weight[weight_idx];
                                }
                            }
                        }
                    }
                    temp_sum += bias[c];
                    temp_sum *= scale[c];
                    val = 1.0f / (1.0f + expf(-temp_sum));
                }
                float diff = val - mean;
                var += diff * diff;
            }
        }
    }
    var /= (groups_per_channel * spatial_size);

    // Normalize
    float eps = 1e-5f;
    float inv_std = rsqrtf(var + eps);
    output[idx] = (sum - mean) * inv_std;
}

torch::Tensor fused_ops_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor scale,
    int num_groups
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    int H_out = height - kernel_size + 1;
    int W_out = width - kernel_size + 1;

    auto output = torch::empty({batch_size, out_channels, H_out, W_out}, input.options());

    int total_elements = batch_size * out_channels * H_out * W_out;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_conv_bias_scale_sigmoid_groupnorm_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        scale.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        num_groups,
        H_out,
        W_out
    );

    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_ops_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor scale,
    int num_groups
);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.num_groups = num_groups
        self.fused_ops = fused_ops

    def forward(self, x):
        # Use fused custom CUDA operator
        return self.fused_ops.fused_ops_cuda(
            x,
            self.conv.weight,
            self.bias,
            self.scale,
            self.num_groups
        )