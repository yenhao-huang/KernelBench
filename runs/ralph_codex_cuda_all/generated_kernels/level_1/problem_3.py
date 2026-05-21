import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

bmm_cpp_source = r"""
torch::Tensor bmm_cuda(torch::Tensor A, torch::Tensor B);
"""

bmm_cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE 16

__global__ void bmm_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size,
    int M,
    int K,
    int N
) {
    __shared__ float As[TILE][TILE + 1];
    __shared__ float Bs[TILE][TILE + 1];

    int batch = blockIdx.z;
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    const float* Ab = A + batch * M * K;
    const float* Bb = B + batch * K * N;
    float* Cb = C + batch * M * N;

    float acc = 0.0f;

    for (int t = 0; t < K; t += TILE) {
        int a_col = t + threadIdx.x;
        int b_row = t + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? Ab[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? Bb[b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; ++i) {
            acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        Cb[row * N + col] = acc;
    }
}

torch::Tensor bmm_cuda(torch::Tensor A, torch::Tensor B) {
    int batch_size = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);

    auto C = torch::empty({batch_size, M, N}, A.options());

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE, batch_size);

    bmm_tiled_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size,
        M,
        K,
        N
    );

    return C;
}
"""

_bmm_ext = load_inline(
    name="kernelbench_bmm_tiled_fp32",
    cpp_sources=bmm_cpp_source,
    cuda_sources=bmm_cuda_source,
    functions=["bmm_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.bmm_ext = _bmm_ext

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.bmm_ext.bmm_cuda(A, B)