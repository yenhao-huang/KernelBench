import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# C++ header (only declarations, no implementations)
cpp_source = """
#include <torch/extension.h>

std::vector<torch::Tensor> compute_groupnorm_stats_cuda(
    torch::Tensor x, int groups, float eps);

torch::Tensor fused_groupnorm_tanh_hardswish_residual_cuda(
    torch::Tensor x,
    torch::Tensor mean,
    torch::Tensor inv_std,
    int groups);

torch::Tensor logsumexp_channel_cuda(torch::Tensor x);
"""

# CUDA source (kernels + wrappers + pybind)
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void compute_groupnorm_stats_kernel_v2(
    const float* __restrict__ x,
    float* __restrict__ mean_out,
    float* __restrict__ inv_std_out,
    int N, int C, int H, int W, int groups, float eps)
{
    int C_per_group = C / groups;
    int total_elements = C_per_group * H * W;

    int sample_idx = blockIdx.x / groups;
    int group_idx = blockIdx.x % groups;
    int group_start_c = group_idx * C_per_group;

    __shared__ float sum_shared[256];
    __shared__ float sum_sq_shared[256];

    float thread_sum = 0.0f;
    float thread_sum_sq = 0.0f;

    for (int idx = threadIdx.x; idx < total_elements; idx += blockDim.x) {
        int c_offset = idx / (H * W);
        int spatial_idx = idx % (H * W);
        int h = spatial_idx / W;
        int w = spatial_idx % W;
        int c = group_start_c + c_offset;

        float val = x[((sample_idx * C + c) * H + h) * W + w];
        thread_sum += val;
        thread_sum_sq += val * val;
    }

    sum_shared[threadIdx.x] = thread_sum;
    sum_sq_shared[threadIdx.x] = thread_sum_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sum_shared[threadIdx.x] += sum_shared[threadIdx.x + stride];
            sum_sq_shared[threadIdx.x] += sum_sq_shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float sum = sum_shared[0];
        float sum_sq = sum_sq_shared[0];
        float count = (float)total_elements;
        float mean = sum / count;
        float var = (sum_sq / count) - (mean * mean);
        float inv_std = 1.0f / sqrtf(var + eps);
        int out_idx = sample_idx * groups + group_idx;
        mean_out[out_idx] = mean;
        inv_std_out[out_idx] = inv_std;
    }
}

__global__ void fused_groupnorm_tanh_hardswish_residual_kernel(
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ inv_std,
    float* __restrict__ out,
    int N, int C, int H, int W, int groups)
{
    int total_elements = N * C * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int C_per_group = C / groups;
    int n = idx / (C * H * W);
    int rem = idx % (C * H * W);
    int c = rem / (H * W);
    int spatial = rem % (H * W);
    int h = spatial / W;
    int w = spatial % W;

    int group_idx = c / C_per_group;
    float m = mean[n * groups + group_idx];
    float istd = inv_std[n * groups + group_idx];
    float x_val = x[idx];

    float normed = (x_val - m) * istd;
    float tanh_val = tanhf(normed);

    float hs;
    if (tanh_val <= -3.0f) {
        hs = 0.0f;
    } else if (tanh_val >= 3.0f) {
        hs = tanh_val;
    } else {
        hs = tanh_val * (tanh_val + 3.0f) / 6.0f;
    }

    out[idx] = x_val + hs;
}

__global__ void logsumexp_channel_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int N, int C, int H, int W)
{
    int total_spatial = N * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_spatial) return;

    int n = idx / (H * W);
    int spatial = idx % (H * W);
    int h = spatial / W;
    int w = spatial % W;

    float max_val = -INFINITY;
    for (int c = 0; c < C; ++c) {
        float val = x[((n * C + c) * H + h) * W + w];
        if (val > max_val) max_val = val;
    }

    float sum_exp = 0.0f;
    for (int c = 0; c < C; ++c) {
        float val = x[((n * C + c) * H + h) * W + w];
        sum_exp += expf(val - max_val);
    }

    out[idx] = max_val + logf(sum_exp);
}

std::vector<torch::Tensor> compute_groupnorm_stats_cuda(
    torch::Tensor x, int groups, float eps)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    auto mean = torch::empty({N, groups}, x.options());
    auto inv_std = torch::empty({N, groups}, x.options());

    const int threads = 256;
    const int blocks = N * groups;

    compute_groupnorm_stats_kernel_v2<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        mean.data_ptr<float>(),
        inv_std.data_ptr<float>(),
        N, C, H, W, groups, eps);

    return {mean, inv_std};
}

torch::Tensor fused_groupnorm_tanh_hardswish_residual_cuda(
    torch::Tensor x,
    torch::Tensor mean,
    torch::Tensor inv_std,
    int groups)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(mean.is_cuda(), "mean must be a CUDA tensor");
    TORCH_CHECK(inv_std.is_cuda(), "inv_std must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous() && mean.is_contiguous() && inv_std.is_contiguous(),
                "all inputs must be contiguous");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    auto out = torch::empty_like(x);

    const int threads = 256;
    const int total_elements = N * C * H * W;
    const int blocks = (total_elements + threads - 1) / threads;

    fused_groupnorm_tanh_hardswish_residual_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        mean.data_ptr<float>(),
        inv_std.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W, groups);

    return out;
}

torch::Tensor logsumexp_channel_cuda(torch::Tensor x)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    auto out = torch::empty({N, 1, H, W}, x.options());

    const int threads = 256;
    const int total_spatial = N * H * W;
    const int blocks = (total_spatial + threads - 1) / threads;

    logsumexp_channel_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_groupnorm_stats_cuda", &compute_groupnorm_stats_cuda, "Compute group norm stats (CUDA)");
    m.def("fused_groupnorm_tanh_hardswish_residual_cuda", &fused_groupnorm_tanh_hardswish_residual_cuda, "Fused group norm + tanh + hardswish + residual (CUDA)");
    m.def("logsumexp_channel_cuda", &logsumexp_channel_cuda, "LogSumExp over channels (CUDA)");
}
"""

# Compile the inline CUDA code
custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=[
        "compute_groupnorm_stats_cuda",
        "fused_groupnorm_tanh_hardswish_residual_cuda",
        "logsumexp_channel_cuda",
    ],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.groups = groups
        self.eps = eps
        self.custom_ops = custom_ops

    def forward(self, x):
        # Convolution (keep as standard cuDNN)
        x_conv = self.conv(x)

        # Step 1: compute group norm statistics (mean, inv_std)
        mean, inv_std = self.custom_ops.compute_groupnorm_stats_cuda(
            x_conv, self.groups, self.eps)

        # Step 2: fused GroupNorm + Tanh + HardSwish + residual add
        x_res = self.custom_ops.fused_groupnorm_tanh_hardswish_residual_cuda(
            x_conv, mean, inv_std, self.groups)

        # Step 3: LogSumExp over channels
        x_logsumexp = self.custom_ops.logsumexp_channel_cuda(x_res)

        return x_logsumexp