import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for fused ReLU + GroupNorm
fused_relu_groupnorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

__global__ void compute_sum_sq_kernel(
    const float* __restrict__ x,
    float* __restrict__ sum,
    float* __restrict__ sq_sum,
    int N, int C, int D, int H, int W, int G, int C_per_group,
    int spatial_size)
{
    int g = blockIdx.x;
    int n = blockIdx.y;
    int block_id = blockIdx.z;

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    int spatial_per_block = (spatial_size + gridDim.z - 1) / gridDim.z;
    int spatial_start = block_id * spatial_per_block;
    int spatial_end = min(spatial_start + spatial_per_block, spatial_size);

    extern __shared__ float shared[];
    float* s_sum = shared;
    float* s_sq_sum = shared + block_size;

    s_sum[tid] = 0.0f;
    s_sq_sum[tid] = 0.0f;
    __syncthreads();

    for (int s = spatial_start + tid; s < spatial_end; s += block_size) {
        int w = s % W;
        int h = (s / W) % H;
        int d = s / (H * W);
        for (int c_local = 0; c_local < C_per_group; ++c_local) {
            int c = g * C_per_group + c_local;
            int idx = ((n * C + c) * D + d) * H * W + h * W + w;
            float val = x[idx];
            val = fmaxf(0.0f, val);
            s_sum[tid] += val;
            s_sq_sum[tid] += val * val;
        }
    }
    __syncthreads();

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sq_sum[tid] += s_sq_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&sum[n * G + g], s_sum[0]);
        atomicAdd(&sq_sum[n * G + g], s_sq_sum[0]);
    }
}

__global__ void compute_mean_var_kernel(
    const float* __restrict__ sum,
    const float* __restrict__ sq_sum,
    float* __restrict__ mean,
    float* __restrict__ var,
    int N, int G, int count, float eps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * G) {
        float s = sum[idx];
        float sq = sq_sum[idx];
        float m = s / count;
        mean[idx] = m;
        var[idx] = fmaxf(sq / count - m * m, 0.0f);
    }
}

__global__ void normalize_kernel(
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    int N, int C, int D, int H, int W, int G, int C_per_group,
    float eps)
{
    int total_elements = N * C * D * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        int w = idx % W;
        int h = (idx / W) % H;
        int d = (idx / (W * H)) % D;
        int c = (idx / (W * H * D)) % C;
        int n = idx / (W * H * D * C);

        int g = c / C_per_group;
        int mean_var_idx = n * G + g;

        float val = x[idx];
        val = fmaxf(0.0f, val);
        float m = mean[mean_var_idx];
        float v = var[mean_var_idx];
        float inv_std = rsqrtf(v + eps);
        float norm_val = (val - m) * inv_std;
        y[idx] = norm_val * gamma[c] + beta[c];
    }
}

torch::Tensor fused_relu_groupnorm_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    int num_groups,
    float eps)
{
    int N = x.size(0);
    int C = x.size(1);
    int D = x.size(2);
    int H = x.size(3);
    int W = x.size(4);
    int G = num_groups;
    int C_per_group = C / G;
    int spatial_size = D * H * W;
    int count = C_per_group * spatial_size;

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto sum = torch::zeros({N, G}, options);
    auto sq_sum = torch::zeros({N, G}, options);
    auto mean = torch::empty({N, G}, options);
    auto var = torch::empty({N, G}, options);
    auto y = torch::empty_like(x);

    int threads = 256;
    int num_blocks_spatial = std::min(128, (spatial_size + 31) / 32);
    dim3 grid(G, N, num_blocks_spatial);
    size_t shared_mem = 2 * threads * sizeof(float);
    compute_sum_sq_kernel<<<grid, threads, shared_mem>>>(
        x.data_ptr<float>(), sum.data_ptr<float>(), sq_sum.data_ptr<float>(),
        N, C, D, H, W, G, C_per_group, spatial_size);

    int total_groups = N * G;
    int threads2 = 256;
    int blocks2 = (total_groups + threads2 - 1) / threads2;
    compute_mean_var_kernel<<<blocks2, threads2>>>(
        sum.data_ptr<float>(), sq_sum.data_ptr<float>(),
        mean.data_ptr<float>(), var.data_ptr<float>(),
        N, G, count, eps);

    int total_elements = N * C * D * H * W;
    int threads3 = 256;
    int blocks3 = (total_elements + threads3 - 1) / threads3;
    normalize_kernel<<<blocks3, threads3>>>(
        x.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(),
        gamma.data_ptr<float>(), beta.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C, D, H, W, G, C_per_group, eps);

    return y;
}
"""

fused_relu_groupnorm_cpp_source = (
    "torch::Tensor fused_relu_groupnorm_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, int num_groups, float eps);"
)

# Compile the inline CUDA code
fused_relu_groupnorm = load_inline(
    name="fused_relu_groupnorm",
    cpp_sources=fused_relu_groupnorm_cpp_source,
    cuda_sources=fused_relu_groupnorm_source,
    functions=["fused_relu_groupnorm_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias)
        self.num_groups = groups
        self.eps = 1e-5
        # Learnable parameters for GroupNorm
        self.gamma = nn.Parameter(torch.ones(out_channels))
        self.beta = nn.Parameter(torch.zeros(out_channels))
        self.fused_op = fused_relu_groupnorm

    def forward(self, x):
        x = self.conv_transpose(x)
        # Fused ReLU + GroupNorm
        x = self.fused_op.fused_relu_groupnorm_cuda(x, self.gamma, self.beta, self.num_groups, self.eps)
        return x