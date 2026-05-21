import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void convt_bn_tanh_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ conv_b,
    const float* __restrict__ bn_w,
    const float* __restrict__ bn_b,
    const float* __restrict__ bn_mean,
    const float* __restrict__ bn_var,
    float* __restrict__ y,
    int N, int Cin, int Hin, int Win, int Cout, int K, int pad, int Hout, int Wout,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * Cout * Hout * Wout;
    if (idx >= total) return;

    int ow = idx % Wout;
    int oh = (idx / Wout) % Hout;
    int co = (idx / (Wout * Hout)) % Cout;
    int n = idx / (Wout * Hout * Cout);

    float acc = conv_b ? conv_b[co] : 0.0f;

    #pragma unroll
    for (int ci = 0; ci < Cin; ++ci) {
        for (int kh = 0; kh < K; ++kh) {
            int ih = oh + pad - kh;
            if ((unsigned)ih >= (unsigned)Hin) continue;
            for (int kw = 0; kw < K; ++kw) {
                int iw = ow + pad - kw;
                if ((unsigned)iw >= (unsigned)Win) continue;
                float xv = x[((n * Cin + ci) * Hin + ih) * Win + iw];
                float wv = w[((ci * Cout + co) * K + kh) * K + kw];
                acc += xv * wv;
            }
        }
    }

    float norm = (acc - bn_mean[co]) * rsqrtf(bn_var[co] + eps);
    float v = norm * bn_w[co] + bn_b[co];
    y[idx] = tanhf(v);
}

__global__ void pool2x2_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int total, int C, int Hin, int Win, int Hout, int Wout
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int ow = idx % Wout;
    int oh = (idx / Wout) % Hout;
    int c = (idx / (Wout * Hout)) % C;
    int n = idx / (Wout * Hout * C);

    int ih = oh * 2;
    int iw = ow * 2;
    int base = ((n * C + c) * Hin + ih) * Win + iw;

    float m = x[base];
    m = fmaxf(m, x[base + 1]);
    m = fmaxf(m, x[base + Win]);
    m = fmaxf(m, x[base + Win + 1]);
    y[idx] = m;
}

__global__ void group_stats_kernel(
    const float* __restrict__ x,
    float* __restrict__ mean,
    float* __restrict__ invstd,
    int N, int C, int H, int W, int G, float eps
) {
    int idx = blockIdx.x;
    int n = idx / G;
    int g = idx % G;
    int cpg = C / G;
    int elems = cpg * H * W;

    float sum = 0.0f;
    float sumsq = 0.0f;

    for (int i = threadIdx.x; i < elems; i += blockDim.x) {
        int t = i;
        int w = t % W;
        t /= W;
        int h = t % H;
        int c = g * cpg + (t / H);
        float v = x[((n * C + c) * H + h) * W + w];
        sum += v;
        sumsq += v * v;
    }

    __shared__ float ssum[256];
    __shared__ float ssq[256];
    ssum[threadIdx.x] = sum;
    ssq[threadIdx.x] = sumsq;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            ssum[threadIdx.x] += ssum[threadIdx.x + s];
            ssq[threadIdx.x] += ssq[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float m = ssum[0] / elems;
        float var = ssq[0] / elems - m * m;
        mean[idx] = m;
        invstd[idx] = rsqrtf(var + eps);
    }
}

__global__ void group_norm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ invstd,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    int total, int C, int H, int W, int G
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int c = (idx / (W * H)) % C;
    int n = idx / (W * H * C);
    int g = c / (C / G);
    int sg = n * G + g;

    float v = (x[idx] - mean[sg]) * invstd[sg];
    y[idx] = v * gamma[c] + beta[c];
}

torch::Tensor convt_bn_tanh_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor conv_b,
    torch::Tensor bn_w,
    torch::Tensor bn_b,
    torch::Tensor bn_mean,
    torch::Tensor bn_var,
    int64_t pad,
    double eps
) {
    int N = x.size(0), Cin = x.size(1), Hin = x.size(2), Win = x.size(3);
    int Cout = w.size(1), K = w.size(2);
    int Hout = Hin - 2 * (int)pad + K;
    int Wout = Win - 2 * (int)pad + K;

    auto y = torch::empty({N, Cout, Hout, Wout}, x.options());
    int total = N * Cout * Hout * Wout;
    int threads = 128;
    int blocks = (total + threads - 1) / threads;

    convt_bn_tanh_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), conv_b.data_ptr<float>(),
        bn_w.data_ptr<float>(), bn_b.data_ptr<float>(),
        bn_mean.data_ptr<float>(), bn_var.data_ptr<float>(),
        y.data_ptr<float>(), N, Cin, Hin, Win, Cout, K, (int)pad, Hout, Wout, (float)eps
    );
    return y;
}

torch::Tensor pool_group_norm_cuda(
    torch::Tensor x,
    torch::Tensor gn_w,
    torch::Tensor gn_b,
    int64_t groups,
    double eps
) {
    int N = x.size(0), C = x.size(1), Hin = x.size(2), Win = x.size(3);
    int H = Hin / 2, W = Win / 2;
    auto pooled = torch::empty({N, C, H, W}, x.options());
    auto y = torch::empty_like(pooled);

    int total = N * C * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    pool2x2_kernel<<<blocks, threads>>>(x.data_ptr<float>(), pooled.data_ptr<float>(), total, C, Hin, Win, H, W);

    auto mean = torch::empty({N, groups}, x.options());
    auto invstd = torch::empty({N, groups}, x.options());
    group_stats_kernel<<<N * groups, 256>>>(pooled.data_ptr<float>(), mean.data_ptr<float>(), invstd.data_ptr<float>(), N, C, H, W, (int)groups, (float)eps);
    group_norm_kernel<<<blocks, threads>>>(pooled.data_ptr<float>(), mean.data_ptr<float>(), invstd.data_ptr<float>(), gn_w.data_ptr<float>(), gn_b.data_ptr<float>(), y.data_ptr<float>(), total, C, H, W, (int)groups);

    return y;
}
"""

cpp_sources = r"""
torch::Tensor convt_bn_tanh_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor conv_b,
    torch::Tensor bn_w,
    torch::Tensor bn_b,
    torch::Tensor bn_mean,
    torch::Tensor bn_var,
    int64_t pad,
    double eps
);

torch::Tensor pool_group_norm_cuda(
    torch::Tensor x,
    torch::Tensor gn_w,
    torch::Tensor gn_b,
    int64_t groups,
    double eps
);
"""

kb_ops = load_inline(
    name="kb_convt_bn_tanh_pool_gn",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["convt_bn_tanh_cuda", "pool_group_norm_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.padding = padding
        self.num_groups = num_groups

    def forward(self, x):
        y = kb_ops.convt_bn_tanh_cuda(
            x.contiguous(),
            self.conv_transpose.weight.contiguous(),
            self.conv_transpose.bias.contiguous(),
            self.batch_norm.weight.contiguous(),
            self.batch_norm.bias.contiguous(),
            self.batch_norm.running_mean.contiguous(),
            self.batch_norm.running_var.contiguous(),
            self.padding,
            self.batch_norm.eps,
        )
        return kb_ops.pool_group_norm_cuda(
            y,
            self.group_norm.weight.contiguous(),
            self.group_norm.bias.contiguous(),
            self.num_groups,
            self.group_norm.eps,
        )