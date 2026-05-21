import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for fused operation: multiply -> instance norm -> multiply -> clamp -> max over channels
fused_op_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void compute_stats_kernel(
    const float* __restrict__ x,
    const float* __restrict__ multiplier,
    float* __restrict__ mean,
    float* __restrict__ inv_std,
    int N, int C, int D, int H, int W,
    int spatial_size)
{
    int idx = blockIdx.x;
    int n = idx / C;
    int c = idx % C;
    float mult = multiplier[c];
    extern __shared__ float shared[];
    float* s_sum = shared;
    float* s_sum_sq = &shared[blockDim.x];
    int tid = threadIdx.x;
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int flat_idx = n * C * spatial_size + c * spatial_size + i;
        float val = x[flat_idx] * mult;
        sum += val;
        sum_sq += val * val;
    }
    s_sum[tid] = sum;
    s_sum_sq[tid] = sum_sq;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sum_sq[tid] += s_sum_sq[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        float total_sum = s_sum[0];
        float total_sum_sq = s_sum_sq[0];
        float mean_val = total_sum / spatial_size;
        float var_val = total_sum_sq / spatial_size - mean_val * mean_val;
        const float eps = 1e-5f;
        float inv_std_val = rsqrtf(var_val + eps);
        mean[idx] = mean_val;
        inv_std[idx] = inv_std_val;
    }
}

__global__ void fused_second_kernel(
    const float* __restrict__ x,
    const float* __restrict__ multiplier,
    const float* __restrict__ mean,
    const float* __restrict__ inv_std,
    const float* __restrict__ inst_weight,
    const float* __restrict__ inst_bias,
    float clamp_min, float clamp_max,
    float* __restrict__ out,
    int N, int C, int D, int H, int W,
    int spatial_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_spatial = N * D * H * W;
    if (idx >= total_spatial) return;
    int n = idx / (D * H * W);
    int rem = idx % (D * H * W);
    int d = rem / (H * W);
    rem = rem % (H * W);
    int h = rem / W;
    int w = rem % W;
    float max_val = -INFINITY;
    for (int c = 0; c < C; ++c) {
        int x_idx = n * C * spatial_size + c * spatial_size + d * H * W + h * W + w;
        float val = x[x_idx] * multiplier[c];
        val = (val - mean[n * C + c]) * inv_std[n * C + c];
        val = val * inst_weight[c] + inst_bias[c];
        val = val * multiplier[c];
        val = fminf(fmaxf(val, clamp_min), clamp_max);
        if (val > max_val) max_val = val;
    }
    out[idx] = max_val;
}

torch::Tensor fused_op_cuda(
    torch::Tensor x,
    torch::Tensor multiplier,
    torch::Tensor inst_weight,
    torch::Tensor inst_bias,
    float clamp_min,
    float clamp_max)
{
    int N = x.size(0);
    int C = x.size(1);
    int D = x.size(2);
    int H = x.size(3);
    int W = x.size(4);
    int spatial_size = D * H * W;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto mean = torch::empty({N, C}, options);
    auto inv_std = torch::empty({N, C}, options);
    int threads_stats = 256;
    int blocks_stats = N * C;
    int shared_mem_size = 2 * threads_stats * sizeof(float);
    compute_stats_kernel<<<blocks_stats, threads_stats, shared_mem_size>>>(
        x.data_ptr<float>(), multiplier.data_ptr<float>(),
        mean.data_ptr<float>(), inv_std.data_ptr<float>(),
        N, C, D, H, W, spatial_size);
    auto out = torch::empty({N, D, H, W}, options);
    int total_spatial = N * spatial_size;
    int threads_second = 256;
    int blocks_second = (total_spatial + threads_second - 1) / threads_second;
    fused_second_kernel<<<blocks_second, threads_second>>>(
        x.data_ptr<float>(), multiplier.data_ptr<float>(),
        mean.data_ptr<float>(), inv_std.data_ptr<float>(),
        inst_weight.data_ptr<float>(), inst_bias.data_ptr<float>(),
        clamp_min, clamp_max,
        out.data_ptr<float>(),
        N, C, D, H, W, spatial_size);
    return out;
}
"""

fused_op_cpp_source = "torch::Tensor fused_op_cuda(torch::Tensor x, torch::Tensor multiplier, torch::Tensor inst_weight, torch::Tensor inst_bias, float clamp_min, float clamp_max);"

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_cuda_source,
    functions=["fused_op_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    A 3D convolutional layer followed by a fused CUDA operator that performs:
    multiply, instance normalization, clamp, multiply, and max over channels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.instance_norm = nn.InstanceNorm3d(out_channels)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv(x)
        # Prepare 1D contiguous multiplier for the kernel
        mult_1d = self.multiplier.view(-1).contiguous()
        inst_weight = self.instance_norm.weight
        inst_bias = self.instance_norm.bias
        x = self.fused_op.fused_op_cuda(x, mult_1d, inst_weight, inst_bias, self.clamp_min, self.clamp_max)
        return x


# The following functions are kept unchanged from the original architecture
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
multiplier_shape = (out_channels, 1, 1, 1)
clamp_min = -1.0
clamp_max = 1.0


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max]