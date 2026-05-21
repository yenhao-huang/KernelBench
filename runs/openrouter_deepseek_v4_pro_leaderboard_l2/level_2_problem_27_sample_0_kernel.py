import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA sources
compute_stats_hardswish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ float hardswish(float x) {
    if (x <= -3.0f) return 0.0f;
    if (x >= 3.0f) return x;
    return x * (x + 3.0f) / 6.0f;
}

__global__ void compute_stats_kernel(
    const float* __restrict__ conv_out,
    float* __restrict__ stats,   // (B, num_groups, 2)  [..., 0] mean, [..., 1] var
    int B, int C, int D, int H, int W,
    int num_groups,
    int group_size,          // per_channel_elems * channels_per_group
    int channels_per_group,
    int per_channel_elems,
    int blocks_per_group) {

    int batch = blockIdx.x / num_groups;
    int group = blockIdx.x % num_groups;
    int block_in_group = blockIdx.y;   // 0 ... blocks_per_group-1

    extern __shared__ float sdata[];  // size: 2 * blockDim.x, first half for sum, second for sumSq
    float* sum_arr = sdata;
    float* sumSq_arr = sdata + blockDim.x;

    int tid = threadIdx.x;
    int stride = blockDim.x * blocks_per_group;
    int base_offset = batch * (C * D * H * W) + group * channels_per_group * per_channel_elems;

    float local_sum = 0.0f;
    float local_sumSq = 0.0f;

    // Loop over elements assigned to this thread
    for (int i = block_in_group * blockDim.x + tid; i < group_size; i += stride) {
        // i is linear index within the group (flattened over channels in group and spatial dims)
        // map to channel offset and spatial offset
        int c_in_group = i / per_channel_elems;
        int spatial_idx = i % per_channel_elems;
        int offset = base_offset + c_in_group * per_channel_elems + spatial_idx;
        float val = conv_out[offset];
        val = hardswish(val);
        local_sum += val;
        local_sumSq += val * val;
    }

    sum_arr[tid] = local_sum;
    sumSq_arr[tid] = local_sumSq;
    __syncthreads();

    // Reduction within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sum_arr[tid] += sum_arr[tid + s];
            sumSq_arr[tid] += sumSq_arr[tid + s];
        }
        __syncthreads();
    }

    // Write block results atomically to global stats
    if (tid == 0) {
        int stats_idx = batch * num_groups * 2 + group * 2; // mean at +0, var at +1 (variance accumulated)
        atomicAdd(&stats[stats_idx], sum_arr[0]);
        atomicAdd(&stats[stats_idx + 1], sumSq_arr[0]);
    }
}

torch::Tensor compute_group_stats_hardswish_cuda(
    torch::Tensor conv_out,
    int num_groups) {

    const auto B = conv_out.size(0);
    const auto C = conv_out.size(1);
    const auto D = conv_out.size(2);
    const auto H = conv_out.size(3);
    const auto W = conv_out.size(4);

    int channels_per_group = C / num_groups;
    int per_channel_elems = D * H * W;
    int group_size = channels_per_group * per_channel_elems;

    // Allocate stats: (B, num_groups, 2) zeroed
    auto stats = torch::zeros({B, num_groups, 2}, conv_out.options());

    const int threads = 256;
    int blocks_per_group = (group_size + threads * 4 - 1) / (threads * 4); // 4 items per thread
    // We'll fix items per thread in kernel dynamically using stride. blocks_per_group is minimum number of blocks needed to cover group_size with stride blockDim * blocks_per_group.
    // Actually, we need any number of blocks, we'll use ceil(group_size / (threads * items_per_thread)), but let's just set items_per_thread = 1 and compute blocks_per_group = (group_size + threads - 1) / threads.
    // To be safe, we use 1 element per thread, so blocks_per_group = ceil(group_size / threads).
    blocks_per_group = (group_size + threads - 1) / threads;

    dim3 grid(B * num_groups, blocks_per_group);
    dim3 block(threads);

    size_t shared_mem = 2 * threads * sizeof(float);

    compute_stats_kernel<<<grid, block, shared_mem>>>(
        conv_out.data_ptr<float>(),
        stats.data_ptr<float>(),
        B, C, D, H, W,
        num_groups,
        group_size,
        channels_per_group,
        per_channel_elems,
        blocks_per_group
    );

    // After kernel, stats contain sum and sumSq per group. Convert to mean and variance (population variance)
    // We'll do a small post-processing on GPU? Or we can incorporate division inside a separate small kernel.
    // Since stats is small, we can do it in a quick second kernel or on CPU. Here we'll do it in a tiny GPU kernel.
    // We'll launch a tiny kernel to finalize stats.
    auto final_stats = torch::empty_like(stats);
    // For simplicity, we'll do the division on CPU after waiting for kernel? But easier: just do a separate small kernel.
    // But we can include a finalization step in the same function: call a simple kernel.
    // Actually, we can just compute mean and var in the same kernel after reduction, but we already used atomics. We'll compute final stats in a small kernel.

    // We'll add another kernel to finalize.
    // To keep it within one function, we can call a small kernel. We'll define it inside as well.
    // But to keep the example focused, we can compute mean / var from sums on CPU after syncing? That's fine because stats is tiny.
    cudaDeviceSynchronize();
    // Now compute mean = sum/group_size, var = sumSq/group_size - mean^2
    // We'll do it on CPU.
    auto stats_cpu = stats.cpu();
    auto final_stats_cpu = torch::empty_like(stats_cpu);
    auto stats_acc = stats_cpu.accessor<float, 3>();
    auto final_acc = final_stats_cpu.accessor<float, 3>();
    for (int b = 0; b < B; ++b) {
        for (int g = 0; g < num_groups; ++g) {
            float sum = stats_acc[b][g][0];
            float sum_sq = stats_acc[b][g][1];
            float mean = sum / group_size;
            float var = sum_sq / group_size - mean * mean;
            final_acc[b][g][0] = mean;
            final_acc[b][g][1] = var;
        }
    }
    auto final_stats_gpu = final_stats_cpu.to(conv_out.device());
    return final_stats_gpu;
}
"""

compute_stats_hardswish_cpp = "torch::Tensor compute_group_stats_hardswish_cuda(torch::Tensor conv_out, int num_groups);"

norm_spatial_mean_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ float hardswish(float x) {
    if (x <= -3.0f) return 0.0f;
    if (x >= 3.0f) return x;
    return x * (x + 3.0f) / 6.0f;
}

__global__ void norm_spatial_mean_kernel(
    const float* __restrict__ conv_out,
    const float* __restrict__ stats,   // (B, num_groups, 2)
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,        // (B, C)
    float eps,
    int B, int C, int D, int H, int W,
    int num_groups) {

    int linear_idx = blockIdx.x; // 0..B*C-1
    int b = linear_idx / C;
    int c = linear_idx % C;
    int channels_per_group = C / num_groups;
    int group = c / channels_per_group;

    // Load stats for this batch and group
    float mean = stats[b * num_groups * 2 + group * 2];
    float var = stats[b * num_groups * 2 + group * 2 + 1];
    float inv_std = 1.0f / sqrtf(var + eps);

    float gamma_val = gamma[c];
    float beta_val = beta[c];

    int base_offset = b * (C * D * H * W) + c * (D * H * W);
    int spatial_size = D * H * W;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    float sum = 0.0f;
    for (int i = tid; i < spatial_size; i += stride) {
        float x = conv_out[base_offset + i];
        x = hardswish(x);
        float normed = (x - mean) * inv_std * gamma_val + beta_val;
        sum += normed;
    }

    // Block reduction
    extern __shared__ float sdata[];
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        output[b * C + c] = sdata[0] / spatial_size;
    }
}

torch::Tensor group_norm_spatial_mean_cuda(
    torch::Tensor conv_out,
    torch::Tensor stats,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps,
    int num_groups) {

    const auto B = conv_out.size(0);
    const auto C = conv_out.size(1);
    const auto D = conv_out.size(2);
    const auto H = conv_out.size(3);
    const auto W = conv_out.size(4);

    auto output = torch::empty({B, C}, conv_out.options());

    int threads = 256;
    dim3 grid(B * C);
    dim3 block(threads);
    size_t shared_mem = threads * sizeof(float);

    norm_spatial_mean_kernel<<<grid, block, shared_mem>>>(
        conv_out.data_ptr<float>(),
        stats.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        eps,
        B, C, D, H, W,
        num_groups
    );

    return output;
}
"""

norm_spatial_mean_cpp = (
    "torch::Tensor group_norm_spatial_mean_cuda("
    "torch::Tensor conv_out, "
    "torch::Tensor stats, "
    "torch::Tensor gamma, "
    "torch::Tensor beta, "
    "float eps, "
    "int num_groups);"
)

# Compile the custom CUDA extensions
compute_stats_hardswish = load_inline(
    name="compute_stats_hardswish",
    cpp_sources=compute_stats_hardswish_cpp,
    cuda_sources=compute_stats_hardswish_source,
    functions=["compute_group_stats_hardswish_cuda"],
    verbose=False,
    extra_cflags=[],
    extra_ldflags=[],
)

norm_spatial_mean = load_inline(
    name="norm_spatial_mean",
    cpp_sources=norm_spatial_mean_cpp,
    cuda_sources=norm_spatial_mean_source,
    functions=["group_norm_spatial_mean_cuda"],
    verbose=False,
    extra_cflags=[],
    extra_ldflags=[],
)


class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernels fusing HardSwish, GroupNorm, and spatial mean.
    Conv3d remains unchanged (cuDNN).
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.num_groups = num_groups
        self.group_norm = nn.GroupNorm(num_groups, out_channels)  # kept for gamma/beta parameters
        self.compute_stats_hardswish = compute_stats_hardswish
        self.norm_spatial_mean = norm_spatial_mean

    def forward(self, x):
        # 1. Conv3D
        conv_out = self.conv(x)                                 # (B, C, D, H, W)
        # 2. Compute group statistics after HardSwish (fused)
        stats = self.compute_stats_hardswish.compute_group_stats_hardswish_cuda(
            conv_out, self.num_groups
        )                                                       # (B, num_groups, 2)
        # 3. Fused HardSwish, GroupNorm normalization, and spatial mean
        out = self.norm_spatial_mean.group_norm_spatial_mean_cuda(
            conv_out,
            stats,
            self.group_norm.weight,    # gamma
            self.group_norm.bias,      # beta
            self.group_norm.eps,
            self.num_groups
        )                                                       # (B, C)
        return out