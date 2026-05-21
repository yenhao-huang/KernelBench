import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for fused batch normalization + scaling
fused_bn_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel to compute per-channel mean and variance
__global__ void compute_mean_var_kernel(
    const float* __restrict__ input,
    float* __restrict__ mean,
    float* __restrict__ var,
    int N, int C, int H, int W,
    int spatial_size)
{
    extern __shared__ float shared[];
    float* s_sum = shared;
    float* s_sum_sq = shared + blockDim.x;

    int c = blockIdx.x;
    if (c >= C) return;

    int tid = threadIdx.x;
    int total_elements = N * spatial_size;

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    // Grid-stride loop over all elements in this channel
    for (int i = tid; i < total_elements; i += blockDim.x) {
        int n = i / spatial_size;
        int spatial_idx = i % spatial_size;
        int h = spatial_idx / W;
        int w = spatial_idx % W;
        int idx = ((n * C + c) * H + h) * W + w;
        float val = input[idx];
        local_sum += val;
        local_sum_sq += val * val;
    }

    s_sum[tid] = local_sum;
    s_sum_sq[tid] = local_sum_sq;
    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sum_sq[tid] += s_sum_sq[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float sum = s_sum[0];
        float sum_sq = s_sum_sq[0];
        float m = sum / total_elements;
        mean[c] = m;
        var[c] = sum_sq / total_elements - m * m;
    }
}

// Kernel to apply normalization, affine, and scaling
__global__ void bn_scale_kernel(
    const float* __restrict__ input,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float eps,
    float scaling_factor,
    float* __restrict__ output,
    int N, int C, int H, int W)
{
    int total_elements = N * C * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int spatial_size = H * W;
    int n = idx / (C * spatial_size);
    int rem = idx % (C * spatial_size);
    int c = rem / spatial_size;
    int spatial_idx = rem % spatial_size;
    int h = spatial_idx / W;
    int w = spatial_idx % W;

    int in_idx = ((n * C + c) * H + h) * W + w;
    float x = input[in_idx];
    float m = mean[c];
    float v = var[c];
    float inv_std = rsqrtf(v + eps);
    float normalized = (x - m) * inv_std;
    float out_val = weight[c] * normalized + bias[c];
    out_val *= scaling_factor;
    output[in_idx] = out_val;
}

torch::Tensor fused_bn_scale_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps,
    double scaling_factor)
{
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int spatial_size = H * W;

    auto mean = torch::empty({C}, input.options());
    auto var = torch::empty({C}, input.options());
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int shared_mem_size = 2 * block_size * sizeof(float);

    // Compute mean and variance
    compute_mean_var_kernel<<<C, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        N, C, H, W, spatial_size);

    // Apply batch norm + scaling
    int total_elements = N * C * H * W;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    bn_scale_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        static_cast<float>(eps),
        static_cast<float>(scaling_factor),
        output.data_ptr<float>(),
        N, C, H, W);

    return output;
}
"""

fused_bn_scale_cpp_source = (
    "torch::Tensor fused_bn_scale_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, double eps, double scaling_factor);"
)

# Compile the inline CUDA code
fused_bn_scale = load_inline(
    name="fused_bn_scale",
    cpp_sources=fused_bn_scale_cpp_source,
    cuda_sources=fused_bn_scale_source,
    functions=["fused_bn_scale_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor
        self.fused_bn_scale = fused_bn_scale

    def forward(self, x):
        x = self.conv(x)
        # Fused batch norm + scaling using custom CUDA kernel
        x = self.fused_bn_scale.fused_bn_scale_cuda(
            x,
            self.bn.weight,
            self.bn.bias,
            self.bn.eps,
            self.scaling_factor
        )
        return x