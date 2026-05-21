import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for fused softmax, subtract, swish, and max reduction
fused_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_kernel(const float* __restrict__ x, const float* __restrict__ subtract_param, float* __restrict__ out,
                             int batch, int channels, int depth, int height, int width) {
    // Each warp processes one spatial location (b,d,h,w)
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int warps_per_block = blockDim.x / 32;
    int global_warp_id = blockIdx.x * warps_per_block + warp_id;
    int total_spatial = depth * height * width;
    int N = batch * total_spatial;
    if (global_warp_id >= N) return;

    int b = global_warp_id / total_spatial;
    int spatial_idx = global_warp_id % total_spatial;
    int d = spatial_idx / (height * width);
    int hw = spatial_idx % (height * width);
    int h = hw / width;
    int w = hw % width;

    // Base offset for this spatial location (without channel)
    int base_spatial = b * (channels * total_spatial) + d * (height * width) + h * width + w;
    int channel_stride = total_spatial;

    float val;
    if (lane_id < channels) {
        val = x[base_spatial + lane_id * channel_stride];
    } else {
        val = -INFINITY;
    }

    // Softmax: find max
    float max_val = val;
    for (int offset = 16; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    max_val = __shfl_sync(0xffffffff, max_val, 0);

    // Compute exp and sum
    float exp_val = (lane_id < channels) ? expf(val - max_val) : 0.0f;
    float sum_exp = exp_val;
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
    }
    sum_exp = __shfl_sync(0xffffffff, sum_exp, 0);

    // Softmax value
    float softmax_val = exp_val / sum_exp;

    // Subtract parameter (only for valid channels)
    float sub_val = (lane_id < channels) ? subtract_param[lane_id] : 0.0f;
    float after_sub = softmax_val - sub_val;

    // Swish: x * sigmoid(x) = x / (1 + exp(-x))
    float swish_val;
    if (lane_id < channels) {
        swish_val = after_sub / (1.0f + expf(-after_sub));
    } else {
        swish_val = -INFINITY;
    }

    // Max reduction across channels
    float max_swish = swish_val;
    for (int offset = 16; offset > 0; offset /= 2) {
        max_swish = fmaxf(max_swish, __shfl_down_sync(0xffffffff, max_swish, offset));
    }

    // Lane 0 writes the result
    if (lane_id == 0) {
        out[global_warp_id] = max_swish;
    }
}

torch::Tensor fused_softmax_subtract_swish_max_cuda(torch::Tensor x, torch::Tensor subtract_param) {
    int batch = x.size(0);
    int channels = x.size(1);
    int depth = x.size(2);
    int height = x.size(3);
    int width = x.size(4);
    int total_spatial = depth * height * width;
    int N = batch * total_spatial;

    auto out = torch::empty({batch, depth, height, width}, x.options());

    const int threads_per_block = 256; // 8 warps per block
    const int warps_per_block = threads_per_block / 32;
    const int blocks = (N + warps_per_block - 1) / warps_per_block;

    fused_kernel<<<blocks, threads_per_block>>>(x.data_ptr<float>(), subtract_param.data_ptr<float>(), out.data_ptr<float>(),
                                                batch, channels, depth, height, width);

    return out;
}
"""

fused_cpp_source = "torch::Tensor fused_softmax_subtract_swish_max_cuda(torch::Tensor x, torch::Tensor subtract_param);"

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_cuda_source,
    functions=["fused_softmax_subtract_swish_max_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernel fusing Softmax, Subtract, Swish, and Max.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.max_pool = nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        self.subtract = nn.Parameter(torch.randn(out_channels))
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.max_pool(x)
        x = self.fused_op.fused_softmax_subtract_swish_max_cuda(x, self.subtract)
        return x