import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for ConvTranspose2d + GELU + GroupNorm fusion
fused_conv_gelu_gn_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// GELU activation function
__device__ float gelu(float x) {
    return x * 0.5f * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
}

// Fused kernel: ConvTranspose2d + GELU + GroupNorm
__global__ void fused_conv_transpose_gelu_groupnorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ gn_weight,
    const float* __restrict__ gn_bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int groups,
    int num_groups,
    int channels_per_group
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_height * out_width;
    if (idx >= total_elements) return;

    // Compute indices
    int n = idx / (out_channels * out_height * out_width);
    int rem = idx % (out_channels * out_height * out_width);
    int oc = rem / (out_height * out_width);
    rem = rem % (out_height * out_width);
    int oh = rem / out_width;
    int ow = rem % out_width;

    // Determine group for GroupNorm
    int group_idx = oc / channels_per_group;
    int group_start = group_idx * channels_per_group;
    int group_end = group_start + channels_per_group;

    // Compute ConvTranspose2d output for this position
    float val = bias != nullptr ? bias[oc] : 0.0f;
    
    int in_h_start = max(0, (oh - kernel_size + 1 + stride - 1) / stride);
    int in_h_end = min(in_height - 1, oh / stride);
    int in_w_start = max(0, (ow - kernel_size + 1 + stride - 1) / stride);
    int in_w_end = min(in_width - 1, ow / stride);

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int ih = oh - kh;
                int iw = ow - kw;
                if (ih >= 0 && ih < in_height * stride && iw >= 0 && iw < in_width * stride && ih % stride == 0 && iw % stride == 0) {
                    ih /= stride;
                    iw /= stride;
                    if (ih < in_height && iw < in_width) {
                        int input_idx = ((n * in_channels + ic) * in_height + ih) * in_width + iw;
                        int weight_idx = ((ic * out_channels + oc) * kernel_size + kh) * kernel_size + kw;
                        val += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    // Apply GELU
    val = gelu(val);

    // Compute GroupNorm statistics and normalize
    // We need mean and variance for the group this channel belongs to
    // Since we're in a single kernel, we compute these on-the-fly using shared memory
    extern __shared__ float shared_mem[];
    float* group_sum = shared_mem;
    float* group_sq_sum = shared_mem + num_groups;
    float* group_count = shared_mem + 2 * num_groups;

    int tid = threadIdx.x;
    if (tid < num_groups) {
        group_sum[tid] = 0.0f;
        group_sq_sum[tid] = 0.0f;
        group_count[tid] = 0.0f;
    }
    __syncthreads();

    // Accumulate statistics for this group
    atomicAdd(&group_sum[group_idx], val);
    atomicAdd(&group_sq_sum[group_idx], val * val);
    atomicAdd(&group_count[group_idx], 1.0f);
    __syncthreads();

    // Compute mean and variance
    float mean = group_sum[group_idx] / group_count[group_idx];
    float variance = group_sq_sum[group_idx] / group_count[group_idx] - mean * mean;
    float inv_std = rsqrtf(variance + 1e-5f);

    // Normalize
    val = (val - mean) * inv_std;
    
    // Apply GroupNorm weight and bias
    if (gn_weight != nullptr) {
        val = val * gn_weight[oc] + (gn_bias != nullptr ? gn_bias[oc] : 0.0f);
    }

    output[idx] = val;
}

torch::Tensor fused_conv_transpose_gelu_groupnorm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    int kernel_size,
    int stride,
    int groups,
    int num_groups
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    int out_channels = weight.size(1);
    int out_height = (in_height - 1) * stride + kernel_size;
    int out_width = (in_width - 1) * stride + kernel_size;
    int channels_per_group = out_channels / num_groups;

    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, input.options());

    int total_elements = batch_size * out_channels * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    int shared_mem_size = 3 * num_groups * sizeof(float);

    fused_conv_transpose_gelu_groupnorm_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        gn_weight.defined() ? gn_weight.data_ptr<float>() : nullptr,
        gn_bias.defined() ? gn_bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        kernel_size,
        stride,
        groups,
        num_groups,
        channels_per_group
    );

    return output;
}
"""

fused_conv_gelu_gn_cpp_source = (
    "torch::Tensor fused_conv_transpose_gelu_groupnorm_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias, "
    "torch::Tensor gn_weight, torch::Tensor gn_bias, "
    "int kernel_size, int stride, int groups, int num_groups);"
)

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_conv_transpose_gelu_groupnorm",
    cpp_sources=fused_conv_gelu_gn_cpp_source,
    cuda_sources=fused_conv_gelu_gn_source,
    functions=["fused_conv_transpose_gelu_groupnorm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.fused_op = fused_op

    def forward(self, x):
        # Use fused CUDA kernel for ConvTranspose2d + GELU + GroupNorm
        return self.fused_op.fused_conv_transpose_gelu_groupnorm_cuda(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.group_norm.weight,
            self.group_norm.bias,
            self.conv_transpose.kernel_size[0],
            self.conv_transpose.stride[0],
            self.conv_transpose.groups,
            self.group_norm.num_groups
        )