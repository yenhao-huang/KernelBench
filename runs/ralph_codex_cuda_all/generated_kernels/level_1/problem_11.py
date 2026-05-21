import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cpp_sources = """
torch::Tensor tensor_matmul4d_cuda(torch::Tensor A, torch::Tensor B);
"""

cuda_sources = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_M 16
#define TILE_N 16
#define TILE_K 32

__global__ void tensor_matmul4d_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int L,
    int K
) {
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N + 1];

    const int n_tiles = (K + TILE_N - 1) / TILE_N;
    const int tile_id = blockIdx.x;
    const int tile_m = tile_id / n_tiles;
    const int tile_n = tile_id - tile_m * n_tiles;

    const int local_m = threadIdx.y;
    const int local_n = threadIdx.x;
    const int row = tile_m * TILE_M + local_m;
    const int col = tile_n * TILE_N + local_n;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    float acc = 0.0f;

    #pragma unroll
    for (int base_l = 0; base_l < L; base_l += TILE_K) {
        for (int idx = tid; idx < TILE_M * TILE_K; idx += TILE_M * TILE_N) {
            int sm = idx / TILE_K;
            int sl = idx - sm * TILE_K;
            int a_row = tile_m * TILE_M + sm;
            int a_l = base_l + sl;
            As[sm][sl] = (a_row < M && a_l < L) ? A[a_row * L + a_l] : 0.0f;
        }

        for (int idx = tid; idx < TILE_K * TILE_N; idx += TILE_M * TILE_N) {
            int sl = idx / TILE_N;
            int sn = idx - sl * TILE_N;
            int b_l = base_l + sl;
            int b_col = tile_n * TILE_N + sn;
            Bs[sl][sn] = (b_l < L && b_col < K) ? B[b_l * K + b_col] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            acc += As[local_m][kk] * Bs[kk][local_n];
        }

        __syncthreads();
    }

    if (row < M && col < K) {
        C[row * K + col] = acc;
    }
}

torch::Tensor tensor_matmul4d_cuda(torch::Tensor A, torch::Tensor B) {
    const int b = A.size(0);
    const int i = A.size(1);
    const int j = A.size(2);
    const int L = A.size(3);
    const int K = B.size(1);
    const int M = b * i * j;

    auto C = torch::empty({b, i, j, K}, A.options());

    const int n_tiles = (K + TILE_N - 1) / TILE_N;
    const int m_tiles = (M + TILE_M - 1) / TILE_M;
    const dim3 block(TILE_N, TILE_M);
    const dim3 grid(m_tiles * n_tiles);

    tensor_matmul4d_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M,
        L,
        K
    );

    return C;
}
"""

tensor_matmul4d = load_inline(
    name="tensor_matmul4d_inline",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["tensor_matmul4d_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = tensor_matmul4d

    def forward(self, A, B):
        return self.op.tensor_matmul4d_cuda(A, B)