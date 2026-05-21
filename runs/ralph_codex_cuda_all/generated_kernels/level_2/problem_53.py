import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#define BM 16
#define BN 16
#define BK 32

__global__ void fused_linear_scale_hardtanh_gelu_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int M,
    int K,
    int N,
    float scale,
    float ht_min,
    float ht_max
) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BN][BK];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * BM + ty;
    int col = blockIdx.x * BN + tx;

    float acc = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BK) {
        int linear = ty * BN + tx;

        int a_idx0 = linear;
        int a_r0 = a_idx0 / BK;
        int a_c0 = a_idx0 - a_r0 * BK;
        int g_ar0 = blockIdx.y * BM + a_r0;
        int g_ac0 = k0 + a_c0;
        As[a_r0][a_c0] = (g_ar0 < M && g_ac0 < K) ? x[g_ar0 * K + g_ac0] : 0.0f;

        int a_idx1 = linear + BM * BN;
        if (a_idx1 < BM * BK) {
            int a_r1 = a_idx1 / BK;
            int a_c1 = a_idx1 - a_r1 * BK;
            int g_ar1 = blockIdx.y * BM + a_r1;
            int g_ac1 = k0 + a_c1;
            As[a_r1][a_c1] = (g_ar1 < M && g_ac1 < K) ? x[g_ar1 * K + g_ac1] : 0.0f;
        }

        int b_idx0 = linear;
        int b_r0 = b_idx0 / BK;
        int b_c0 = b_idx0 - b_r0 * BK;
        int g_br0 = blockIdx.x * BN + b_r0;
        int g_bc0 = k0 + b_c0;
        Bs[b_r0][b_c0] = (g_br0 < N && g_bc0 < K) ? w[g_br0 * K + g_bc0] : 0.0f;

        int b_idx1 = linear + BM * BN;
        if (b_idx1 < BN * BK) {
            int b_r1 = b_idx1 / BK;
            int b_c1 = b_idx1 - b_r1 * BK;
            int g_br1 = blockIdx.x * BN + b_r1;
            int g_bc1 = k0 + b_c1;
            Bs[b_r1][b_c1] = (g_br1 < N && g_bc1 < K) ? w[g_br1 * K + g_bc1] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            acc += As[ty][kk] * Bs[tx][kk];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        float v = (acc + bias[col]) * scale;
        v = fminf(fmaxf(v, ht_min), ht_max);
        float gelu = 0.5f * v * (1.0f + erff(v * 0.70710678118654752440f));
        out[row * N + col] = gelu;
    }
}

torch::Tensor fused_linear_scale_hardtanh_gelu_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor bias,
    double scale,
    double ht_min,
    double ht_max
) {
    int M = (int)x.size(0);
    int K = (int)x.size(1);
    int N = (int)w.size(0);

    auto out = torch::empty({M, N}, x.options());

    dim3 block(BN, BM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    fused_linear_scale_hardtanh_gelu_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        M,
        K,
        N,
        (float)scale,
        (float)ht_min,
        (float)ht_max
    );

    return out;
}
"""

cpp_sources = r"""
torch::Tensor fused_linear_scale_hardtanh_gelu_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor bias,
    double scale,
    double ht_min,
    double ht_max
);
"""

fused_linear_act = load_inline(
    name="fused_linear_scale_hardtanh_gelu_ext",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["fused_linear_scale_hardtanh_gelu_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scaling_factor = float(scaling_factor)
        self.hardtanh_min = float(hardtanh_min)
        self.hardtanh_max = float(hardtanh_max)
        self.fused_op = fused_linear_act

    def forward(self, x):
        return self.fused_op.fused_linear_scale_hardtanh_gelu_cuda(
            x,
            self.gemm.weight,
            self.gemm.bias,
            self.scaling_factor,
            self.hardtanh_min,
            self.hardtanh_max,
        )