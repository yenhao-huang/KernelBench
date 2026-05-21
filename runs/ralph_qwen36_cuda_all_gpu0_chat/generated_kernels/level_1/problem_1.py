import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
# We use a tiled approach to optimize memory access patterns and utilize shared memory.
# This implementation is optimized for FP32.
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Block size for the tile in shared memory
#define TILE_SIZE 16

__global__ void matmul_kernel(const float* A, const float* B, float* C, int N) {
    // Shared memory for tiles of A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread indices within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // The row and column in C that this thread is responsible for
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    // Loop over the tiles needed to compute the dot product
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < num_tiles; ++t) {
        // Load tile from A into shared memory
        // Check bounds to avoid out-of-bounds access if N is not a multiple of TILE_SIZE
        if (row < N && (t * TILE_SIZE + tx) < N) {
            As[ty][tx] = A[row * N + (t * TILE_SIZE + tx)];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load tile from B into shared memory
        if ((t * TILE_SIZE + ty) < N && col < N) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        // Synchronize to ensure all threads have loaded their data
        __syncthreads();

        // Compute partial dot product for this tile
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        // Synchronize before loading the next tile
        __syncthreads();
    }

    // Write the result to global memory
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    auto N = A.size(0);
    auto out = torch::zeros({N, N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    const int block_size = TILE_SIZE;
    dim3 threads(block_size, block_size);
    
    // Calculate grid dimensions
    int blocks_x = (N + block_size - 1) / block_size;
    int blocks_y = (N + block_size - 1) / block_size;
    dim3 blocks(blocks_x, blocks_y);

    matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), out.data_ptr<float>(), N);

    return out;
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
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a single square matrix multiplication (C = A * B)
    using a custom CUDA kernel with shared memory tiling.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_op = matmul_op

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication using the custom CUDA kernel.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        return self.matmul_op.matmul_cuda(A, B)