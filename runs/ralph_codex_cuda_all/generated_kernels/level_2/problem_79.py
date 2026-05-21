import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cpp_sources = """
torch::Tensor fused_conv3d_instnorm_clamp_max_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor multiplier,
    double clamp_min,
    double clamp_max);
"""

cuda_sources = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void conv3d_mul_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    const float* __restrict__ m,
    float* __restrict__ y,
    int N, int Cin, int Cout,
    int D, int H, int Wd,
    int K, int Do, int Ho, int Wo,
    int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int ow = idx % Wo;
    int t = idx / Wo;
    int oh = t % Ho;
    t /= Ho;
    int od = t % Do;
    t /= Do;
    int co = t % Cout;
    int n = t / Cout;

    float acc = b[co];
    for (int ci = 0; ci < Cin; ++ci) {
        for (int kd = 0; kd < K; ++kd) {
            int id = od + kd;
            for (int kh = 0; kh < K; ++kh) {
                int ih = oh + kh;
                for (int kw = 0; kw < K; ++kw) {
                    int iw = ow + kw;
                    int xidx = (((n * Cin + ci) * D + id) * H + ih) * Wd + iw;
                    int widx = (((co * Cin + ci) * K + kd) * K + kh) * K + kw;
                    acc += x[xidx] * w[widx];
                }
            }
        }
    }
    y[idx] = acc * m[co];
}

__global__ void instnorm_clamp_mul_kernel(
    const float* __restrict__ y,
    const float* __restrict__ m,
    float* __restrict__ z,
    int N, int Cout, int S,
    float clamp_min, float clamp_max) {
    int nc = blockIdx.x;
    int n = nc / Cout;
    int c = nc % Cout;
    int base = (n * Cout + c) * S;

    __shared__ float smem[256];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < S; i += blockDim.x) {
        sum += y[base + i];
    }
    smem[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    float mean = smem[0] / (float)S;

    float ss = 0.0f;
    for (int i = threadIdx.x; i < S; i += blockDim.x) {
        float d = y[base + i] - mean;
        ss += d * d;
    }
    smem[threadIdx.x] = ss;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }

    float inv_std = rsqrtf(smem[0] / (float)S + 1.0e-5f);
    float mult = m[c];

    for (int i = threadIdx.x; i < S; i += blockDim.x) {
        float v = (y[base + i] - mean) * inv_std;
        v = fminf(fmaxf(v, clamp_min), clamp_max);
        z[base + i] = v * mult;
    }
}

__global__ void max_channel_kernel(
    const float* __restrict__ z,
    float* __restrict__ out,
    int N, int Cout, int S,
    int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int n = idx / S;
    int s = idx % S;
    float mx = -FLT_MAX;

    for (int c = 0; c < Cout; ++c) {
        float v = z[(n * Cout + c) * S + s];
        mx = fmaxf(mx, v);
    }
    out[idx] = mx;
}

torch::Tensor fused_conv3d_instnorm_clamp_max_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor multiplier,
    double clamp_min,
    double clamp_max) {
    x = x.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();
    multiplier = multiplier.contiguous();

    int N = (int)x.size(0);
    int Cin = (int)x.size(1);
    int D = (int)x.size(2);
    int H = (int)x.size(3);
    int Wd = (int)x.size(4);
    int Cout = (int)weight.size(0);
    int K = (int)weight.size(2);

    int Do = D - K + 1;
    int Ho = H - K + 1;
    int Wo = Wd - K + 1;
    int S = Do * Ho * Wo;

    auto y = torch::empty({N, Cout, Do, Ho, Wo}, x.options());
    auto z = torch::empty_like(y);
    auto out = torch::empty({N, Do, Ho, Wo}, x.options());

    int conv_total = N * Cout * S;
    int out_total = N * S;
    const int threads = 256;

    conv3d_mul_kernel<<<(conv_total + threads - 1) / threads, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        multiplier.data_ptr<float>(),
        y.data_ptr<float>(),
        N, Cin, Cout, D, H, Wd, K, Do, Ho, Wo, conv_total);

    instnorm_clamp_mul_kernel<<<N * Cout, threads>>>(
        y.data_ptr<float>(),
        multiplier.data_ptr<float>(),
        z.data_ptr<float>(),
        N, Cout, S,
        (float)clamp_min,
        (float)clamp_max);

    max_channel_kernel<<<(out_total + threads - 1) / threads, threads>>>(
        z.data_ptr<float>(),
        out.data_ptr<float>(),
        N, Cout, S, out_total);

    return out;
}
"""

fused_op = load_inline(
    name="kb_fused_conv3d_instnorm_clamp_max",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["fused_conv3d_instnorm_clamp_max_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        return fused_op.fused_conv3d_instnorm_clamp_max_cuda(
            x,
            self.conv.weight,
            self.conv.bias,
            self.multiplier,
            self.clamp_min,
            self.clamp_max,
        )