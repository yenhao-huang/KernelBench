import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_cpp_source = """
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

matmul_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4

__global__ void sgemm_64x64_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int M, int K, int N) {
    __shared__ float As[BM][BK + 1];
    __shared__ float Bs[BK][BN + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    const int block_m = blockIdx.y * BM;
    const int block_n = blockIdx.x * BN;

    const int local_m = ty * TM;
    const int local_n = tx * TN;

    float acc[TM][TN];

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (int k0 = 0; k0 < K; k0 += BK) {
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            int idx = tid + l * 256;

            int a_row = idx / BK;
            int a_col = idx % BK;
            int g_a_row = block_m + a_row;
            int g_a_col = k0 + a_col;
            As[a_row][a_col] = (g_a_row < M && g_a_col < K) ? A[g_a_row * K + g_a_col] : 0.0f;

            int b_row = idx / BN;
            int b_col = idx % BN;
            int g_b_row = k0 + b_row;
            int g_b_col = block_n + b_col;
            Bs[b_row][b_col] = (g_b_row < K && g_b_col < N) ? B[g_b_row * N + g_b_col] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float a_frag[TM];
            float b_frag[TN];

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                a_frag[i] = As[local_m + i][kk];
            }

            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                b_frag[j] = Bs[kk][local_n + j];
            }

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int row = block_m + local_m + i;
        if (row < M) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                int col = block_n + local_n + j;
                if (col < N) {
                    C[row * N + col] = acc[i][j];
                }
            }
        }
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    dim3 block(16, 16);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    sgemm_64x64_kernel<<<grid, block>>>(A.data_ptr<float>(),
                                        B.data_ptr<float>(),
                                        C.data_ptr<float>(),
                                        M, K, N);

    return C;
}
"""

matmul_ext = load_inline(
    name="custom_sgemm_64x64_ext",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_cuda_source,
    functions=["matmul_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_ext = matmul_ext

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_ext.matmul_cuda(A, B)