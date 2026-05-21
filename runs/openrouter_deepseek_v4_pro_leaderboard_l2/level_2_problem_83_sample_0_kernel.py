import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for fused GroupNorm + min + clamp + dropout
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define EPS 1e-5f
#define BLOCK_SIZE 256

// Kernel 1: compute sum and sum of squares per group
__global__ void compute_group_stats_kernel(
    const float* __restrict__ x,
    float* __restrict__ sum,
    float* __restrict__ sum_sq,
    int N, int C, int D, int H, int W, int G,
    int C_per_group, int spatial_size, int group_size,
    int num_blocks_per_group)
{
    extern __shared__ float s_data[];
    float* s_sum = s_data;
    float* s_sum_sq = &s_data[blockDim.x];

    int tid = threadIdx.x;
    int block_idx = blockIdx.x;
    int n = block_idx / (G * num_blocks_per_group);
    int rem = block_idx % (G * num_blocks_per_group);
    int g = rem / num_blocks_per_group;
    int block_in_group = rem % num_blocks_per_group;

    int start = block_in_group * blockDim.x;
    int stride = blockDim.x;

    // Each thread processes multiple elements if group_size > blockDim.x * num_blocks_per_group
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    int base_idx = n * (C * spatial_size) + g * C_per_group * spatial_size;
    for (int i = start + tid; i < group_size; i += stride * num_blocks_per_group) {
        // i is index within the group (flattened channels and spatial)
        int c_in_group = i / spatial_size;
        int spatial_idx = i % spatial_size;
        int c = g * C_per_group + c_in_group;
        int idx = base_idx + c_in_group * spatial_size + spatial_idx;
        float val = x[idx];
        local_sum += val;
        local_sum_sq += val * val;
    }

    s_sum[tid] = local_sum;
    s_sum_sq[tid] = local_sum_sq;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sum_sq[tid] += s_sum_sq[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        int out_idx = n * G + g;
        atomicAdd(&sum[out_idx], s_sum[0]);
        atomicAdd(&sum_sq[out_idx], s_sum_sq[0]);
    }
}

// Kernel 2: normalize, min, clamp, dropout
__global__ void fused_norm_min_clamp_dropout_kernel(
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int N, int C, int D, int H, int W, int G,
    int C_per_group, int spatial_size,
    float min_value, float max_value, float dropout_p,
    bool training, unsigned long long seed, unsigned long long offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * D * H * W;
    if (idx >= total_elements) return;

    // Decode index
    int n = idx / (C * spatial_size);
    int rem = idx % (C * spatial_size);
    int c = rem / spatial_size;
    int spatial_idx = rem % spatial_size;

    int g = c / C_per_group;
    int mean_var_idx = n * G + g;
    float mean_val = mean[mean_var_idx];
    float var_val = var[mean_var_idx];

    float x_val = x[idx];
    float norm_val = (x_val - mean_val) * rsqrtf(var_val + EPS);
    float out_val = norm_val * weight[c] + bias[c];

    // torch.min(x, min_value)
    out_val = fminf(out_val, min_value);
    // torch.clamp(x, min=min_value, max=max_value)
    out_val = fminf(fmaxf(out_val, min_value), max_value);

    // Dropout
    if (training) {
        curandStatePhilox4_32_10_t state;
        curand_init(seed, idx + offset, 0, &state);
        float rand = curand_uniform(&state);
        if (rand < dropout_p) {
            out_val = 0.0f;
        } else {
            out_val = out_val / (1.0f - dropout_p);
        }
    }

    y[idx] = out_val;
}

torch::Tensor fused_groupnorm_min_clamp_dropout_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int groups,
    float min_value,
    float max_value,
    float dropout_p,
    bool training,
    unsigned long long seed)
{
    const auto N = x.size(0);
    const auto C = x.size(1);
    const auto D = x.size(2);
    const auto H = x.size(3);
    const auto W = x.size(4);
    const int G = groups;
    const int C_per_group = C / G;
    const int spatial_size = D * H * W;
    const int group_size = C_per_group * spatial_size;
    const int total_elements = N * C * spatial_size;

    auto y = torch::empty_like(x);

    // Allocate temporary storage for sum and sum_sq per group
    auto sum = torch::zeros({N, G}, torch::device(x.device()).dtype(torch::kFloat32));
    auto sum_sq = torch::zeros({N, G}, torch::device(x.device()).dtype(torch::kFloat32));

    // Kernel 1: compute stats
    const int num_blocks_per_group = (group_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int total_blocks = N * G * num_blocks_per_group;
    const int shared_mem_size = 2 * BLOCK_SIZE * sizeof(float);

    compute_group_stats_kernel<<<total_blocks, BLOCK_SIZE, shared_mem_size>>>(
        x.data_ptr<float>(),
        sum.data_ptr<float>(),
        sum_sq.data_ptr<float>(),
        N, C, D, H, W, G,
        C_per_group, spatial_size, group_size,
        num_blocks_per_group);

    // Compute mean and variance from sum and sum_sq
    auto count = (float)group_size;
    auto mean = sum / count;
    auto var = sum_sq / count - mean * mean;
    // Ensure non-negative variance
    var = torch::clamp_min(var, 0.0f);

    // Kernel 2: fused norm + min + clamp + dropout
    const int total_blocks2 = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    fused_norm_min_clamp_dropout_kernel<<<total_blocks2, BLOCK_SIZE>>>(
        x.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C, D, H, W, G,
        C_per_group, spatial_size,
        min_value, max_value, dropout_p,
        training, seed, 0);

    return y;
}
"""

fused_op_cpp_source = """
torch::Tensor fused_groupnorm_min_clamp_dropout_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int groups,
    float min_value,
    float max_value,
    float dropout_p,
    bool training,
    unsigned long long seed);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_groupnorm_min_clamp_dropout",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_groupnorm_min_clamp_dropout_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class FusedGroupNormMinClampDropout(nn.Module):
    def __init__(self, num_channels, groups, min_value, max_value, dropout_p):
        super().__init__()
        self.groups = groups
        self.min_value = min_value
        self.max_value = max_value
        self.dropout_p = dropout_p
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.seed_counter = 0

    def forward(self, x):
        # Increment seed for each forward to get different dropout masks
        seed = self.seed_counter
        self.seed_counter += 1
        return fused_op.fused_groupnorm_min_clamp_dropout_cuda(
            x, self.weight, self.bias, self.groups,
            self.min_value, self.max_value, self.dropout_p,
            self.training, seed)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.fused_op = FusedGroupNormMinClampDropout(out_channels, groups, min_value, max_value, dropout_p)

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_op(x)
        return x