import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication (C = A^T * B)
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Tile size
    const int TILE_SIZE = 16;

    // Starting indices for this tile
    int row_start = by * TILE_SIZE;
    int col_start = bx * TILE_SIZE;

    // Shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Accumulator for this thread's output element
    float sum = 0.0f;

    // Loop over K tiles
    for (int k_tile = 0; k_tile < K; k_tile += TILE_SIZE) {
        // Load A tile: A is (K, M), we need A[k][row] for k in [k_tile, k_tile+TILE_SIZE) and row in [row_start, row_start+TILE_SIZE)
        // Thread (tx, ty) loads A[k_tile + ty][row_start + tx] if within bounds
        if (k_tile + ty < K && row_start + tx < M) {
            As[ty][tx] = A[(k_tile + ty) * M + (row_start + tx)];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load B tile: B is (K, N), we need B[k][col] for k in [k_tile, k_tile+TILE_SIZE) and col in [col_start, col_start+TILE_SIZE)
        // Thread (tx, ty) loads B[k_tile + ty][col_start + tx] if within bounds
        if (k_tile + ty < K && col_start + tx < N) {
            Bs[ty][tx] = B[(k_tile + ty) * N + (col_start + tx)];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[k][ty] * Bs[k][tx];
        }

        __syncthreads();
    }

    // Write output
    int row = row_start + ty;
    int col = col_start + tx;
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // A: (K, M), B: (K, N)
    int K = A.size(0);
    int M = A.size(1);
    int N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());

    const int TILE_SIZE = 16;
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

    return C;
}
"""

matmul_cpp_source = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code for matrix multiplication
matmul_op = load_inline(
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
        self.matmul_cuda = matmul_op

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using custom CUDA kernel.

        Args:
            A: Input tensor of shape (K, M).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        return self.matmul_cuda.matmul_cuda(A, B)