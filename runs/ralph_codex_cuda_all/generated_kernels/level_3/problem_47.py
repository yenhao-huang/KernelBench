import math
import torch
import torch.nn as nn
import torch as th
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void assign_kernel(
    const float* __restrict__ x,
    const float* __restrict__ clusters,
    const float* __restrict__ bn_w,
    const float* __restrict__ bn_b,
    const float* __restrict__ bn_mean,
    const float* __restrict__ bn_var,
    float* __restrict__ assign,
    int rows, int D, int C, float eps
) {
    int r = blockIdx.x;
    if (r >= rows) return;

    float vals[64];
    float mx = -3.402823466e+38f;

    for (int c = 0; c < C; ++c) {
        float acc = 0.0f;
        const float* xp = x + r * D;
        const float* cp = clusters + c;
        for (int d = 0; d < D; ++d) {
            acc += xp[d] * cp[d * C];
        }
        acc = (acc - bn_mean[c]) * rsqrtf(bn_var[c] + eps) * bn_w[c] + bn_b[c];
        vals[c] = acc;
        mx = fmaxf(mx, acc);
    }

    float denom = 0.0f;
    for (int c = 0; c < C; ++c) {
        vals[c] = expf(vals[c] - mx);
        denom += vals[c];
    }

    float inv = 1.0f / denom;
    float* ap = assign + r * C;
    for (int c = 0; c < C; ++c) {
        ap[c] = vals[c] * inv;
    }
}

__global__ void vlad_kernel(
    const float* __restrict__ x,
    const float* __restrict__ assign,
    const float* __restrict__ clusters2,
    float* __restrict__ out,
    int B, int N, int D, int K
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * D * K;
    if (idx >= total) return;

    int k = idx % K;
    int d = (idx / K) % D;
    int b = idx / (D * K);

    float sum_ax = 0.0f;
    float sum_a = 0.0f;

    for (int n = 0; n < N; ++n) {
        float a = assign[(b * N + n) * K + k];
        sum_a += a;
        sum_ax += a * x[(b * N + n) * D + d];
    }

    out[idx] = sum_ax - sum_a * clusters2[d * K + k];
}

__global__ void intra_norm_kernel(float* __restrict__ vlad, int B, int D, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * K;
    if (idx >= total) return;

    int b = idx / K;
    int k = idx % K;

    float ss = 0.0f;
    for (int d = 0; d < D; ++d) {
        float v = vlad[(b * D + d) * K + k];
        ss += v * v;
    }

    float inv = rsqrtf(ss + 1.0e-12f);
    for (int d = 0; d < D; ++d) {
        vlad[(b * D + d) * K + k] *= inv;
    }
}

__global__ void final_norm_kernel(float* __restrict__ vlad, int B, int DK) {
    int b = blockIdx.x;
    if (b >= B) return;

    float ss = 0.0f;
    for (int i = threadIdx.x; i < DK; i += blockDim.x) {
        float v = vlad[b * DK + i];
        ss += v * v;
    }

    __shared__ float buf[256];
    buf[threadIdx.x] = ss;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }

    float inv = rsqrtf(buf[0] + 1.0e-12f);
    for (int i = threadIdx.x; i < DK; i += blockDim.x) {
        vlad[b * DK + i] *= inv;
    }
}

torch::Tensor netvlad_forward_cuda(
    torch::Tensor x,
    torch::Tensor clusters,
    torch::Tensor bn_w,
    torch::Tensor bn_b,
    torch::Tensor bn_mean,
    torch::Tensor bn_var,
    torch::Tensor clusters2,
    double eps
) {
    int B = x.size(0);
    int N = x.size(1);
    int D = x.size(2);
    int K = clusters2.size(2);
    int C = clusters.size(1);

    auto assign = torch::empty({B * N, C}, x.options());
    auto out = torch::empty({B, D, K}, x.options());

    assign_kernel<<<B * N, 1>>>(
        x.data_ptr<float>(),
        clusters.data_ptr<float>(),
        bn_w.data_ptr<float>(),
        bn_b.data_ptr<float>(),
        bn_mean.data_ptr<float>(),
        bn_var.data_ptr<float>(),
        assign.data_ptr<float>(),
        B * N, D, C, (float)eps
    );

    int total = B * D * K;
    vlad_kernel<<<(total + 255) / 256, 256>>>(
        x.data_ptr<float>(),
        assign.data_ptr<float>(),
        clusters2.data_ptr<float>(),
        out.data_ptr<float>(),
        B, N, D, K
    );

    intra_norm_kernel<<<(B * K + 255) / 256, 256>>>(out.data_ptr<float>(), B, D, K);
    final_norm_kernel<<<B, 256>>>(out.data_ptr<float>(), B, D * K);

    return out.reshape({B, D * K});
}
"""

cpp_sources = "torch::Tensor netvlad_forward_cuda(torch::Tensor x, torch::Tensor clusters, torch::Tensor bn_w, torch::Tensor bn_b, torch::Tensor bn_mean, torch::Tensor bn_var, torch::Tensor clusters2, double eps);"

netvlad_ext = load_inline(
    name="netvlad_inline_ext",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["netvlad_forward_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self, cluster_size, feature_size, ghost_clusters):
        super(ModelNew, self).__init__()
        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.ghost_clusters = ghost_clusters

        init_sc = 1 / math.sqrt(feature_size)
        clusters = cluster_size + ghost_clusters

        self.clusters = nn.Parameter(init_sc * th.randn(feature_size, clusters))
        self.batch_norm = nn.BatchNorm1d(clusters)
        self.clusters2 = nn.Parameter(init_sc * th.randn(1, feature_size, cluster_size))
        self.out_dim = self.cluster_size * feature_size
        self.netvlad_ext = netvlad_ext

    def forward(self, x, mask=None):
        return self.netvlad_ext.netvlad_forward_cuda(
            x.contiguous(),
            self.clusters.contiguous(),
            self.batch_norm.weight.contiguous(),
            self.batch_norm.bias.contiguous(),
            self.batch_norm.running_mean.contiguous(),
            self.batch_norm.running_var.contiguous(),
            self.clusters2.contiguous(),
            self.batch_norm.eps,
        )