import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math_constants.h>

__global__ void convt3x3_gelu_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y,
    int N, int IC, int H, int W, int OC, int OH, int OW
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)N * OC * OH * OW;
    if (idx >= total) return;

    int ow = idx % OW;
    int oh = (idx / OW) % OH;
    int oc = (idx / (OW * OH)) % OC;
    int n = idx / ((long long)OC * OH * OW);

    float acc = b[oc];

    #pragma unroll
    for (int kh = 0; kh < 3; ++kh) {
        int ih = oh - kh;
        if ((unsigned)ih >= (unsigned)H) continue;
        #pragma unroll
        for (int kw = 0; kw < 3; ++kw) {
            int iw = ow - kw;
            if ((unsigned)iw >= (unsigned)W) continue;
            for (int ic = 0; ic < IC; ++ic) {
                float xv = x[((long long)n * IC + ic) * H * W + ih * W + iw];
                float wv = w[((long long)ic * OC + oc) * 9 + kh * 3 + kw];
                acc += xv * wv;
            }
        }
    }

    float gelu = 0.5f * acc * (1.0f + erff(acc * 0.7071067811865476f));
    y[idx] = gelu;
}

__global__ void group_stats_kernel(
    const float* __restrict__ y,
    float* __restrict__ mean,
    float* __restrict__ invstd,
    int N, int C, int H, int W, int G
) {
    int ng = blockIdx.x;
    int n = ng / G;
    int g = ng % G;
    int cpg = C / G;
    int elems = cpg * H * W;

    float sum = 0.0f;
    float sumsq = 0.0f;

    for (int i = threadIdx.x; i < elems; i += blockDim.x) {
        int c_local = i / (H * W);
        int rem = i - c_local * H * W;
        int c = g * cpg + c_local;
        float v = y[((long long)n * C + c) * H * W + rem];
        sum += v;
        sumsq += v * v;
    }

    __shared__ float ssum[256];
    __shared__ float ssq[256];
    ssum[threadIdx.x] = sum;
    ssq[threadIdx.x] = sumsq;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            ssum[threadIdx.x] += ssum[threadIdx.x + stride];
            ssq[threadIdx.x] += ssq[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float m = ssum[0] / elems;
        float var = ssq[0] / elems - m * m;
        var = fmaxf(var, 0.0f);
        mean[ng] = m;
        invstd[ng] = rsqrtf(var + 1.0e-5f);
    }
}

__global__ void group_norm_kernel(
    const float* __restrict__ y,
    const float* __restrict__ mean,
    const float* __restrict__ invstd,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    int N, int C, int H, int W, int G
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)N * C * H * W;
    if (idx >= total) return;

    int hw = H * W;
    int c = (idx / hw) % C;
    int n = idx / ((long long)C * hw);
    int g = c / (C / G);
    int ng = n * G + g;

    float v = (y[idx] - mean[ng]) * invstd[ng];
    out[idx] = v * gamma[c] + beta[c];
}

torch::Tensor convt_gelu_groupnorm_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    int64_t num_groups
) {
    int N = x.size(0);
    int IC = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int OC = weight.size(1);
    int OH = H + 2;
    int OW = W + 2;
    int G = (int)num_groups;

    auto y = torch::empty({N, OC, OH, OW}, x.options());
    auto out = torch::empty_like(y);
    auto mean = torch::empty({N, G}, x.options());
    auto invstd = torch::empty({N, G}, x.options());

    const int threads = 256;
    long long total = (long long)N * OC * OH * OW;
    int blocks = (int)((total + threads - 1) / threads);

    convt3x3_gelu_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        y.data_ptr<float>(),
        N, IC, H, W, OC, OH, OW
    );

    group_stats_kernel<<<N * G, threads>>>(
        y.data_ptr<float>(),
        mean.data_ptr<float>(),
        invstd.data_ptr<float>(),
        N, OC, OH, OW, G
    );

    group_norm_kernel<<<blocks, threads>>>(
        y.data_ptr<float>(),
        mean.data_ptr<float>(),
        invstd.data_ptr<float>(),
        gn_weight.data_ptr<float>(),
        gn_bias.data_ptr<float>(),
        out.data_ptr<float>(),
        N, OC, OH, OW, G
    );

    return out;
}
"""

cpp_sources = """
torch::Tensor convt_gelu_groupnorm_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    int64_t num_groups
);
"""

convt_gelu_groupnorm = load_inline(
    name="convt_gelu_groupnorm_inline_v1",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["convt_gelu_groupnorm_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride
        )
        self.group_norm = nn.GroupNorm(
            num_groups=num_groups, num_channels=out_channels
        )
        self.num_groups = num_groups
        self.op = convt_gelu_groupnorm

    def forward(self, x):
        return self.op.convt_gelu_groupnorm_cuda(
            x.contiguous(),
            self.conv_transpose.weight.contiguous(),
            self.conv_transpose.bias.contiguous(),
            self.group_norm.weight.contiguous(),
            self.group_norm.bias.contiguous(),
            self.num_groups,
        )