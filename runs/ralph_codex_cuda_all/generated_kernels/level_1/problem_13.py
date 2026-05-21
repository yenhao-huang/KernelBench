import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cpp_sources = """
torch::Tensor symm_matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE 32
#define SUB 16

__global__ void symm_matmul_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int N) {
    __shared__ float As[TILE][TILE + 1];
    __shared__ float Bs[TILE][TILE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row0 = blockIdx.y * TILE + ty;
    int row1 = row0 + SUB;
    int col0 = blockIdx.x * TILE + tx;
    int col1 = col0 + SUB;

    float c00 = 0.0f;
    float c01 = 0.0f;
    float c10 = 0.0f;
    float c11 = 0.0f;

    for (int base = 0; base < N; base += TILE) {
        int k0 = base + tx;
        int k1 = base + tx + SUB;
        int r0 = base + ty;
        int r1 = base + ty + SUB;

        As[ty][tx] = (row0 < N && k0 < N) ? A[row0 * N + k0] : 0.0f;
        As[ty][tx + SUB] = (row0 < N && k1 < N) ? A[row0 * N + k1] : 0.0f;
        As[ty + SUB][tx] = (row1 < N && k0 < N) ? A[row1 * N + k0] : 0.0f;
        As[ty + SUB][tx + SUB] = (row1 < N && k1 < N) ? A[row1 * N + k1] : 0.0f;

        Bs[ty][tx] = (r0 < N && col0 < N) ? B[r0 * N + col0] : 0.0f;
        Bs[ty][tx + SUB] = (r0 < N && col1 < N) ? B[r0 * N + col1] : 0.0f;
        Bs[ty + SUB][tx] = (r1 < N && col0 < N) ? B[r1 * N + col0] : 0.0f;
        Bs[ty + SUB][tx + SUB] = (r1 < N && col1 < N) ? B[r1 * N + col1] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            float a0 = As[ty][k];
            float a1 = As[ty + SUB][k];
            float b0 = Bs[k][tx];
            float b1 = Bs[k][tx + SUB];

            c00 += a0 * b0;
            c01 += a0 * b1;
            c10 += a1 * b0;
            c11 += a1 * b1;
        }

        __syncthreads();
    }

    if (row0 < N && col0 < N) C[row0 * N + col0] = c00;
    if (row0 < N && col1 < N) C[row0 * N + col1] = c01;
    if (row1 < N && col0 < N) C[row1 * N + col0] = c10;
    if (row1 < N && col1 < N) C[row1 * N + col1] = c11;
}

torch::Tensor symm_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 threads(SUB, SUB);
    dim3 blocks((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    symm_matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    return C;
}
"""

symm_matmul_ext = load_inline(
    name="symm_matmul_ext",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["symm_matmul_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.op = symm_matmul_ext

    def forward(self, A, B):
        return self.op.symm_matmul_cuda(A, B)