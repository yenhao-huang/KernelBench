import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cpp_sources = """
torch::Tensor tensor_matrix_mul_cuda(torch::Tensor A, torch::Tensor B);
"""

cuda_sources = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BM 16
#define BN 16
#define BK 32

__global__ void tensor_matrix_mul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int rows,
    int K,
    int L
) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * BM + ty;
    int col = blockIdx.x * BN + tx;
    int tid = ty * BN + tx;

    float acc = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BK) {
        #pragma unroll
        for (int i = tid; i < BM * BK; i += BM * BN) {
            int r = i / BK;
            int k = i - r * BK;
            int gr = blockIdx.y * BM + r;
            int gk = k0 + k;
            As[r][k] = (gr < rows && gk < K) ? A[gr * K + gk] : 0.0f;
        }

        #pragma unroll
        for (int i = tid; i < BK * BN; i += BM * BN) {
            int k = i / BN;
            int c = i - k * BN;
            int gk = k0 + k;
            int gc = blockIdx.x * BN + c;
            Bs[k][c] = (gk < K && gc < L) ? B[gk * L + gc] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < rows && col < L) {
        C[row * L + col] = acc;
    }
}

torch::Tensor tensor_matrix_mul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = (int)A.size(0);
    int M = (int)A.size(1);
    int K = (int)A.size(2);
    int L = (int)B.size(1);
    int rows = N * M;

    auto C = torch::empty({N, M, L}, A.options());

    dim3 block(BN, BM);
    dim3 grid((L + BN - 1) / BN, (rows + BM - 1) / BM);

    tensor_matrix_mul_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        rows,
        K,
        L
    );

    return C;
}
"""

tensor_matrix_mul = load_inline(
    name="tensor_matrix_mul_kernelbench",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["tensor_matrix_mul_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.op = tensor_matrix_mul

    def forward(self, A, B):
        return self.op.tensor_matrix_mul_cuda(A, B)