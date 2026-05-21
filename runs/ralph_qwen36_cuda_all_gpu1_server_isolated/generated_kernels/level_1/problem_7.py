import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for Matrix Multiplication optimized for small K dimension (K=64)
# We use a tiled approach to maximize register usage and minimize global memory traffic.
# Since K is small (64), we can fit the entire row of A and column of B into registers/shared memory efficiently.
# However, for very large M and N, a standard blocked GEMM is best.
# Given M=32768, N=32768, K=64, this is a "wide" matrix multiplication.
# We will implement a simple but efficient tiled GEMM.

matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Block dimensions for the tile
#define BLOCK_M 128
#define BLOCK_N 128
#define BLOCK_K 64 // Since K=64, we can process one full K dimension per block if aligned, or use tiling.
                     // With K=64, a single thread block can compute a BLOCK_M x BLOCK_N tile using shared memory for A and B tiles of size BLOCK_M x BLOCK_K and BLOCK_K x BLOCK_N.

__global__ void matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    // Shared memory for tiles of A and B
    __shared__ float sA[BLOCK_M][BLOCK_K];
    __shared__ float sB[BLOCK_K][BLOCK_N];

    // Thread indices within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Global row and column index for this thread's contribution to C
    int row = by * BLOCK_M + ty;
    int col = bx * BLOCK_N + tx;

    float sum = 0.0f;

    // Loop over K dimension in tiles of size BLOCK_K
    // Since K=64 and BLOCK_K=64, this loop will run exactly once if K is a multiple of BLOCK_K.
    // We make it general for any K.
    for (int k = 0; k < K; k += BLOCK_K) {
        // Load tile from A into shared memory
        // Each thread loads one element from A and one from B
        if (row < M && (k + tx) < K) {
            sA[ty][tx] = A[row * K + k + tx];
        } else {
            sA[ty][tx] = 0.0f;
        }

        if ((k + ty) < K && col < N) {
            sB[ty][tx] = B[(k + ty) * N + col];
        } else {
            sB[ty][tx] = 0.0f;
        }

        // Synchronize to ensure all data is loaded
        __syncthreads();

        // Compute partial dot product for this tile
        for (int t = 0; t < BLOCK_K; ++t) {
            sum += sA[ty][t] * sB[t][tx];
        }

        // Synchronize before loading next tile
        __syncthreads();
    }

    // Write result to global memory
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    auto M = A.size(0);
    auto K = A.size(1);
    auto N = B.size(1);

    // Ensure inputs are contiguous and on CUDA
    if (!A.is_contiguous()) A = A.contiguous();
    if (!B.is_contiguous()) B = B.contiguous();

    auto C = torch::zeros({M, N}, A.options());

    const int block_size_x = 32; // Threads per row in block (tx)
    const int block_size_y = 4;  // Threads per col in block (ty) -> Total threads = 128
    
    // Grid dimensions
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    dim3 block(block_size_x, block_size_y);

    matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

    return C;
}
"""

matmul_cpp_source = (
    "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for matrix multiplication
matmul_op = load_inline(
    name="matmul_custom",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA operator for matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = matmul_op

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using custom CUDA kernel.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        return self.matmul.matmul_cuda(A, B)