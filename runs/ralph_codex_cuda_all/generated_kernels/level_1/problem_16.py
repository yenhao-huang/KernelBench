import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_at_b_cpp = """
torch::Tensor matmul_at_b_cuda(torch::Tensor A, torch::Tensor B);
"""

matmul_at_b_cuda = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BM 16
#define BN 16
#define BK 32

__global__ void matmul_at_b_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int M, int K, int N) {
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * BM + ty;
    int col = blockIdx.x * BN + tx;
    int tid = ty * BN + tx;

    float acc = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BK) {
        #pragma unroll
        for (int i = tid; i < BK * BM; i += BM * BN) {
            int kk = i / BM;
            int mm = i - kk * BM;
            int gk = k0 + kk;
            int gm = blockIdx.y * BM + mm;
            As[kk][mm] = (gk < K && gm < M) ? A[gk * M + gm] : 0.0f;
        }

        #pragma unroll
        for (int i = tid; i < BK * BN; i += BM * BN) {
            int kk = i / BN;
            int nn = i - kk * BN;
            int gk = k0 + kk;
            int gn = blockIdx.x * BN + nn;
            Bs[kk][nn] = (gk < K && gn < N) ? B[gk * N + gn] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            acc += As[kk][ty] * Bs[kk][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

torch::Tensor matmul_at_b_cuda(torch::Tensor A, torch::Tensor B) {
    int K = A.size(0);
    int M = A.size(1);
    int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    dim3 block(BN, BM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    matmul_at_b_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    return C;
}
"""

matmul_at_b = load_inline(
    name="matmul_at_b_inline",
    cpp_sources=matmul_at_b_cpp,
    cuda_sources=matmul_at_b_cuda,
    functions=["matmul_at_b_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_at_b = matmul_at_b

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_at_b.matmul_at_b_cuda(A, B)