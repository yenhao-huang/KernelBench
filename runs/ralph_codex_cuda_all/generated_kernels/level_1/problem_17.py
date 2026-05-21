import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BM 16
#define BN 16
#define BK 32

__global__ void matmul_abt_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K
) {
    __shared__ float As[BM][BK + 1];
    __shared__ float Bs[BN][BK + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * BM + ty;
    const int col = blockIdx.x * BN + tx;

    float acc = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BK) {
        const int linear = ty * BN + tx;

        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int idx = linear + i * 256;

            int a_r = idx / BK;
            int a_k = idx - a_r * BK;
            if (a_r < BM) {
                int gr = blockIdx.y * BM + a_r;
                int gk = k0 + a_k;
                As[a_r][a_k] = (gr < M && gk < K) ? A[gr * K + gk] : 0.0f;
            }

            int b_r = idx / BK;
            int b_k = idx - b_r * BK;
            if (b_r < BN) {
                int gc = blockIdx.x * BN + b_r;
                int gk = k0 + b_k;
                Bs[b_r][b_k] = (gc < N && gk < K) ? B[gc * K + gk] : 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            acc += As[ty][kk] * Bs[tx][kk];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

torch::Tensor matmul_abt_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = (int)A.size(0);
    const int K = (int)A.size(1);
    const int N = (int)B.size(0);

    auto C = torch::empty({M, N}, A.options());

    dim3 block(BN, BM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    matmul_abt_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M,
        N,
        K
    );

    return C;
}
"""

cpp_sources = "torch::Tensor matmul_abt_cuda(torch::Tensor A, torch::Tensor B);"

matmul_abt_ext = load_inline(
    name="kb_matmul_abt_fp32",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["matmul_abt_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_abt = matmul_abt_ext

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_abt.matmul_abt_cuda(A, B)