import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for symmetric matrix multiplication.
# Since A and B are symmetric, C = A * B is not necessarily symmetric, 
# but we can potentially optimize memory access patterns or use specific algorithms.
# However, for general N=4096, standard cuBLAS sgemm is highly optimized.
# To demonstrate a custom operator as requested, we will implement a tiled matrix multiplication
# that leverages shared memory to reduce global memory bandwidth pressure.
# This is a simplified block-based approach suitable for demonstration of custom CUDA integration.

custom_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_SIZE 32

__global__ void symmetric_matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
    // Shared memory for tiles of A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Thread indices within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global row and column indices for the current thread
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    // Loop over tiles
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < num_tiles; ++t) {
        // Load tile from A into shared memory
        // A is accessed by row 'row' and column 't * TILE_SIZE + tx'
        if (row < N && (t * TILE_SIZE + tx) < N) {
            As[ty][tx] = A[row * N + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load tile from B into shared memory
        // B is accessed by row 't * TILE_SIZE + ty' and column 'col'
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

    // Write result to global memory
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor symmetric_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    auto N = A.size(0);
    auto out = torch::zeros({N, N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    const int block_size_x = TILE_SIZE;
    const int block_size_y = TILE_SIZE;
    
    // Calculate grid dimensions
    int grid_x = (N + block_size_x - 1) / block_size_x;
    int grid_y = (N + block_size_y - 1) / block_size_y;

    dim3 block(block_size_x, block_size_y);
    dim3 grid(grid_x, grid_y);

    symmetric_matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), out.data_ptr<float>(), N);

    return out;
}
"""

custom_matmul_cpp_source = (
    "torch::Tensor symmetric_matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for matrix multiplication
symmetric_matmul = load_inline(
    name="symmetric_matmul",
    cpp_sources=custom_matmul_cpp_source,
    cuda_sources=custom_matmul_source,
    functions=["symmetric_matmul_cuda"],
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
    
    def forward(self, A, B):
        return symmetric_matmul.symmetric_matmul_cuda(A, B)