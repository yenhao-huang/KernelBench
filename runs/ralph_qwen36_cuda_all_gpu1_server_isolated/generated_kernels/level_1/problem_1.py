import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
# We use a tiled approach to optimize memory access patterns and utilize shared memory.
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Block size for tiling
#define TILE_SIZE 32

__global__ void matmul_kernel(const float* A, const float* B, float* C, int N) {
    // Each block computes a TILE_SIZE x TILE_SIZE tile of the output matrix C
    // blockIdx.x corresponds to the row tile index in C
    // blockIdx.y corresponds to the column tile index in C
    
    int row = blockIdx.x * TILE_SIZE + threadIdx.y;
    int col = blockIdx.y * TILE_SIZE + threadIdx.x;

    // Shared memory for tiles of A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    // Loop over the tiles needed to compute C(row, col)
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < num_tiles; ++t) {
        // Load tile from A into shared memory
        // A is accessed as A[row * N + k] where k is the column index in the current tile
        int a_row = row;
        int a_col = t * TILE_SIZE + threadIdx.x;
        
        if (a_row < N && a_col < N) {
            As[threadIdx.y][threadIdx.x] = A[a_row * N + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile from B into shared memory
        // B is accessed as B[k * N + col] where k is the row index in the current tile
        int b_row = t * TILE_SIZE + threadIdx.y;
        int b_col = col;
        
        if (b_row < N && b_col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + b_col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Synchronize to ensure all threads have loaded their data
        __syncthreads();

        // Compute partial dot product for this tile
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
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
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D tensors");
    
    int N = A.size(0);
    TORCH_CHECK(A.size(1) == N, "A must be square");
    TORCH_CHECK(B.size(0) == N, "B must be square");
    TORCH_CHECK(B.size(1) == N, "B must be square");

    auto C = torch::zeros({N, N}, A.options());

    const int block_size = TILE_SIZE; // 32x32 threads per block
    dim3 grid((N + block_size - 1) / block_size, (N + block_size - 1) / block_size);
    
    matmul_kernel<<<grid, block_size>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    return C;
}
"""

matmul_cpp_source = (
    "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for matrix multiplication
matmul_op = load_inline(
    name="matmul_op",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"]
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a single square matrix multiplication (C = A * B)
    using a custom tiled CUDA kernel.
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