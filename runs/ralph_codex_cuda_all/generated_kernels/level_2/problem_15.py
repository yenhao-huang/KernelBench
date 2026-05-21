import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

__inline__ __device__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__inline__ __device__ float block_reduce_sum(float v) {
    static __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) shared[wid] = v;
    __syncthreads();
    v = (threadIdx.x < (blockDim.x + 31) / 32) ? shared[lane] : 0.0f;
    if (wid == 0) v = warp_reduce_sum(v);
    return v;
}

__global__ void spatial_mean_kernel(const float* __restrict__ x, float* __restrict__ spatial_mean,
                                    int N, int C, int S) {
    int nc = blockIdx.x;
    int n = nc / C;
    int c = nc - n * C;
    const float* ptr = x + ((n * C + c) * S);

    float sum = 0.0f;
    for (int i = threadIdx.x; i < S; i += blockDim.x) {
        sum += ptr[i];
    }
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        spatial_mean[nc] = sum / (float)S;
    }
}

__global__ void channel_stats_kernel(const float* __restrict__ x,
                                     float* __restrict__ ch_sum,
                                     float* __restrict__ ch_sumsq,
                                     int N, int C, int S) {
    int c = blockIdx.x;
    float sum = 0.0f;
    float sumsq = 0.0f;

    for (int idx = threadIdx.x; idx < N * S; idx += blockDim.x) {
        int n = idx / S;
        int s = idx - n * S;
        float v = x[(n * C + c) * S + s];
        sum += v;
        sumsq += v * v;
    }

    sum = block_reduce_sum(sum);
    __syncthreads();
    sumsq = block_reduce_sum(sumsq);

    if (threadIdx.x == 0) {
        ch_sum[c] = sum;
        ch_sumsq[c] = sumsq;
    }
}

__global__ void update_running_kernel(const float* __restrict__ ch_sum,
                                      const float* __restrict__ ch_sumsq,
                                      float* __restrict__ running_mean,
                                      float* __restrict__ running_var,
                                      int C, int M, float momentum) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < C) {
        float mean = ch_sum[c] / (float)M;
        float var = ch_sumsq[c] / (float)M - mean * mean;
        float unbiased = var * ((float)M / (float)(M - 1));
        running_mean[c] = running_mean[c] * (1.0f - momentum) + mean * momentum;
        running_var[c] = running_var[c] * (1.0f - momentum) + unbiased * momentum;
    }
}

__global__ void centered_bn_kernel(const float* __restrict__ x,
                                   const float* __restrict__ spatial_mean,
                                   const float* __restrict__ ch_sum,
                                   const float* __restrict__ ch_sumsq,
                                   const float* __restrict__ weight,
                                   const float* __restrict__ running_var,
                                   float* __restrict__ out,
                                   int N, int C, int S, int M,
                                   bool training, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * S;
    if (idx >= total) return;

    int s = idx % S;
    int tmp = idx / S;
    int c = tmp % C;
    int n = tmp / C;

    float var;
    if (training) {
        float mean_c = ch_sum[c] / (float)M;
        var = ch_sumsq[c] / (float)M - mean_c * mean_c;
    } else {
        var = running_var[c];
    }

    float scale = weight[c] * rsqrtf(var + eps);
    out[idx] = (x[idx] - spatial_mean[n * C + c]) * scale;
}

torch::Tensor bn_center_cuda(torch::Tensor x,
                             torch::Tensor weight,
                             torch::Tensor running_mean,
                             torch::Tensor running_var,
                             bool training,
                             double momentum,
                             double eps) {
    int N = (int)x.size(0);
    int C = (int)x.size(1);
    int S = (int)(x.size(2) * x.size(3) * x.size(4));
    int M = N * S;

    auto out = torch::empty_like(x);
    auto spatial_mean = torch::empty({N, C}, x.options());
    auto ch_sum = torch::empty({C}, x.options());
    auto ch_sumsq = torch::empty({C}, x.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int reduce_threads = 256;
    spatial_mean_kernel<<<N * C, reduce_threads, 0, stream>>>(
        x.data_ptr<float>(), spatial_mean.data_ptr<float>(), N, C, S
    );

    if (training) {
        channel_stats_kernel<<<C, reduce_threads, 0, stream>>>(
            x.data_ptr<float>(), ch_sum.data_ptr<float>(), ch_sumsq.data_ptr<float>(), N, C, S
        );
        update_running_kernel<<<(C + 255) / 256, 256, 0, stream>>>(
            ch_sum.data_ptr<float>(), ch_sumsq.data_ptr<float>(),
            running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
            C, M, (float)momentum
        );
    }

    int total = N * C * S;
    centered_bn_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        x.data_ptr<float>(), spatial_mean.data_ptr<float>(),
        ch_sum.data_ptr<float>(), ch_sumsq.data_ptr<float>(),
        weight.data_ptr<float>(), running_var.data_ptr<float>(),
        out.data_ptr<float>(), N, C, S, M, training, (float)eps
    );

    return out;
}
"""

cpp_sources = "torch::Tensor bn_center_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor running_mean, torch::Tensor running_var, bool training, double momentum, double eps);"

bn_center_ext = load_inline(
    name="bn_center_ext",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["bn_center_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias
        )
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        momentum = self.batch_norm.momentum
        if self.training and self.batch_norm.track_running_stats:
            self.batch_norm.num_batches_tracked.add_(1)
            if momentum is None:
                momentum = 1.0 / float(self.batch_norm.num_batches_tracked.item())
        if momentum is None:
            momentum = 0.0
        return bn_center_ext.bn_center_cuda(
            x,
            self.batch_norm.weight,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.training,
            float(momentum),
            float(self.batch_norm.eps),
        )