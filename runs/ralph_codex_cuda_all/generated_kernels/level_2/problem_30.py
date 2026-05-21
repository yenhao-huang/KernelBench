import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define TILE 16

__global__ void linear_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y,
    int N, int K, int O
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    for (int t = 0; t < K; t += TILE) {
        int ak = t + threadIdx.x;
        int bk = t + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < N && ak < K) ? x[row * K + ak] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (col < O && bk < K) ? w[col * K + bk] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; ++i) {
            acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < O) {
        y[row * O + col] = acc + b[col];
    }
}

__global__ void groupnorm_hardtanh_kernel(
    const float* __restrict__ inp,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    int N, int C, int G,
    float eps,
    float minv,
    float maxv
) {
    extern __shared__ float smem[];
    float* ssum = smem;
    float* ssq = smem + blockDim.x;

    int ng = blockIdx.x;
    int n = ng / G;
    int g = ng - n * G;
    int group_size = C / G;
    int base = n * C + g * group_size;

    float local_sum = 0.0f;
    float local_sq = 0.0f;

    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float v = inp[base + i];
        local_sum += v;
        local_sq += v * v;
    }

    ssum[threadIdx.x] = local_sum;
    ssq[threadIdx.x] = local_sq;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            ssum[threadIdx.x] += ssum[threadIdx.x + stride];
            ssq[threadIdx.x] += ssq[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float mean = ssum[0] / (float)group_size;
    float var = ssq[0] / (float)group_size - mean * mean;
    float inv_std = rsqrtf(var + eps);

    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        int c = g * group_size + i;
        float v = (inp[base + i] - mean) * inv_std;
        v = v * gamma[c] + beta[c];
        v = fminf(fmaxf(v, minv), maxv);
        out[base + i] = v;
    }
}

torch::Tensor linear_groupnorm_hardtanh_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gamma,
    torch::Tensor beta,
    int64_t num_groups,
    double minv,
    double maxv
) {
    int N = (int)x.size(0);
    int K = (int)x.size(1);
    int O = (int)weight.size(0);

    auto tmp = torch::empty({N, O}, x.options());
    auto out = torch::empty({N, O}, x.options());

    dim3 block(TILE, TILE);
    dim3 grid((O + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    linear_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        tmp.data_ptr<float>(),
        N, K, O
    );

    int threads = 256;
    int groups_total = N * (int)num_groups;
    size_t shmem = threads * 2 * sizeof(float);
    groupnorm_hardtanh_kernel<<<groups_total, threads, shmem>>>(
        tmp.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        out.data_ptr<float>(),
        N, O, (int)num_groups,
        1.0e-5f,
        (float)minv,
        (float)maxv
    );

    return out;
}
"""

cpp_sources = """
torch::Tensor linear_groupnorm_hardtanh_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gamma,
    torch::Tensor beta,
    int64_t num_groups,
    double minv,
    double maxv
);
"""

linear_gn_ht = load_inline(
    name="linear_gn_ht_inline",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["linear_groupnorm_hardtanh_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.num_groups = num_groups
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

    def forward(self, x):
        return linear_gn_ht.linear_groupnorm_hardtanh_cuda(
            x.contiguous(),
            self.gemm.weight.contiguous(),
            self.gemm.bias.contiguous(),
            self.group_norm.weight.contiguous(),
            self.group_norm.bias.contiguous(),
            self.num_groups,
            self.hardtanh_min,
            self.hardtanh_max,
        )