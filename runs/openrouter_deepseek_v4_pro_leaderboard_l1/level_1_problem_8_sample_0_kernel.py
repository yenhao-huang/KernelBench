import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_M 128
#define TILE_N 128
#define TILE_K 8
#define BLOCK_M 16
#define BLOCK_N 16
#define min(a,b) ((a)<(b)?(a):(b))

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;

    int row_start = block_y * TILE_M;
    int col_start = block_x * TILE_N;

    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    int tid = thread_y * blockDim.x + thread_x;

    float sum[8][8] = {0.0f};

    for (int k_block = 0; k_block < K; k_block += TILE_K) {
        // Zero out shared memory
        for (int i = tid; i < TILE_M * TILE_K; i += blockDim.x * blockDim.y) {
            ((float*)As)[i] = 0.0f;
        }
        for (int i = tid; i < TILE_K * TILE_N; i += blockDim.x * blockDim.y) {
            ((float*)Bs)[i] = 0.0f;
        }
        __syncthreads();

        int k_start = k_block;
        int k_end = min(k_start + TILE_K, K);

        // Load A tile
        int a_row = tid / (TILE_K / 4);
        int a_col = (tid % (TILE_K / 4)) * 4;
        int global_row = row_start + a_row;
        int global_col = k_start + a_col;
        if (global_row < M && global_col < k_end) {
            float4 val = *reinterpret_cast<const float4*>(&A[global_row * K + global_col]);
            As[a_row][a_col] = val.x;
            if (a_col + 1 < TILE_K && global_col + 1 < k_end) As[a_row][a_col + 1] = val.y;
            if (a_col + 2 < TILE_K && global_col + 2 < k_end) As[a_row][a_col + 2] = val.z;
            if (a_col + 3 < TILE_K && global_col + 3 < k_end) As[a_row][a_col + 3] = val.w;
        }

        // Load B tile
        int b_row = tid / (TILE_N / 4);
        int b_col = (tid % (TILE_N / 4)) * 4;
        global_row = k_start + b_row;
        global_col = col_start + b_col;
        if (global_row < k_end && global_col < N) {
            float4 val = *reinterpret_cast<const float4*>(&B[global_row * N + global_col]);
            Bs[b_row][b_col] = val.x;
            if (b_col + 1 < TILE_N && global_col + 1 < N) Bs[b_row][b_col + 1] = val.y;
            if (b_col + 2 < TILE_N && global_col + 2 < N) Bs[b_row][b_col + 2] = val.z;
            if (b_col + 3 < TILE_N && global_col + 3 < N) Bs[b_row][b_col + 3] = val.w;
        }

        __syncthreads();

        // Compute partial products
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                float a_val = As[thread_y * 8 + i][k];
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    sum[i][j] += a_val * Bs[k][thread_x * 8 + j];
                }
            }
        }

        __syncthreads();
    }

    // Write results to C
    for (int i = 0; i < 8; ++i) {
        int row = row_start + thread_y * 8 + i;
        if (row < M) {
            for (int j = 0; j < 8; ++j) {
                int col = col_start + thread_x * 8 + j;
                if (col < N) {
                    C[row * N + col] = sum[i][j];
                }
            }
        }
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());

    dim3 block(BLOCK_N, BLOCK_M);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);

    return C;
}
"""

matmul_cpp_source = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code
matmul_cuda_module = load_inline(
    name="matmul_cuda",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_cuda = matmul_cuda_module

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A = A.contiguous()
        B = B.contiguous()
        return self.matmul_cuda.matmul_cuda(A, B)