import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for Batched Matrix Multiplication (BMM)
# Optimized for FP32 precision.
# We use a tiled approach to maximize shared memory usage and reduce global memory bandwidth pressure.
bmm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Block dimensions for the tile
#define TILE_M 16
#define TILE_N 16
#define TILE_K 8 // K dimension is split into chunks of this size

__global__ void bmm_kernel(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ C, 
    int batch_size, 
    int m, 
    int k, 
    int n
) {
    // Each block computes a tile of the output matrix C for one batch item
    // blockIdx.x: batch index
    // blockIdx.y: row index in the output tile (0 to m/TILE_M - 1)
    // blockIdx.z: col index in the output tile (0 to n/TILE_N - 1)

    int batch = blockIdx.x;
    int ty = blockIdx.y;
    int tz = blockIdx.z;

    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;

    // Global coordinates for this thread's contribution within the tile
    int row = ty * TILE_M + thread_y;
    int col = tz * TILE_N + thread_x;

    // Shared memory tiles for A and B
    // __shared__ float sA[TILE_M][TILE_K];
    // __shared__ float sB[TILE_K][TILE_N];
    
    // Flattened shared memory for better coalescing and indexing simplicity
    __shared__ float sA[TILE_M * TILE_K];
    __shared__ float sB[TILE_K * TILE_N];

    float sum = 0.0f;

    // Loop over K dimension in tiles
    for (int t = 0; t < k; t += TILE_K) {
        // Load tile from A into shared memory
        // Each thread loads one element if within bounds
        int a_row_idx = row;
        int a_col_idx = t + thread_x; // Thread x iterates over K in the tile
        
        // We need to be careful with indexing. 
        // Let's map threads: threadIdx.y -> row offset, threadIdx.x -> col offset (in K)
        // Actually, standard tiling:
        // Threads in block: blockDim.x * blockDim.y = TILE_M * TILE_N? No, usually we use 2D grid for tiles.
        // Let's stick to a simpler 1D thread mapping per tile or standard 2D.
        
        // Re-defining thread mapping for clarity:
        // Block (bx, by, bz) -> Batch, TileRow, TileCol
        // Thread (tx, ty) within block:
        // tx: 0..TILE_N-1 (column index in tile)
        // ty: 0..TILE_M-1 (row index in tile)
        
        int local_row = ty;
        int local_col = tx;

        // Load A[batch, row + local_row, t + local_col]
        if (row + local_row < m && t + local_col < k) {
            sA[local_row * TILE_K + local_col] = A[batch * m * k + (row + local_row) * k + (t + local_col)];
        } else {
            sA[local_row * TILE_K + local_col] = 0.0f;
        }

        // Load B[batch, t + local_row, col + local_col]
        if (t + local_row < k && col + local_col < n) {
            sB[local_row * TILE_N + local_col] = B[batch * k * n + (t + local_row) * n + (col + local_col)];
        } else {
            sB[local_row * TILE_N + local_col] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product for this tile of K
        for (int p = 0; p < TILE_K; ++p) {
             // Check bounds for the inner loop to avoid out-of-bounds access if k is not multiple of TILE_K
             if (t + p < k) {
                 sum += sA[local_row * TILE_K + p] * sB[p * TILE_N + local_col];
             }
        }

        __syncthreads();
    }

    // Write result to global memory
    if (row < m && col < n) {
        C[batch * m * n + row * n + col] = sum;
    }
}

torch::Tensor bmm_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "A must be FP32");
    TORCH_CHECK(B.scalar_type() == torch::kFloat32, "B must be FP32");

    auto batch_size = A.size(0);
    auto m = A.size(1);
    auto k = A.size(2);
    auto n = B.size(2);

    TORCH_CHECK(B.size(1) == k, "A's K dimension must match B's K dimension");

    auto C = torch::zeros({batch_size, m, n}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // Grid dimensions
    // We want one block per tile of the output matrix.
    // Tiles are TILE_M x TILE_N.
    int grid_m = (m + TILE_M - 1) / TILE_M;
    int grid_n = (n + TILE_N - 1) / TILE_N;

    dim3 grid(batch_size, grid_m, grid_n);
    
    // Block dimensions: TILE_M rows, TILE_N cols
    dim3 block(TILE_N, TILE_M); // x is col index in tile, y is row index in tile

    bmm_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), batch_size, m, k, n);

    return C;
}
"""

bmm_cpp_source = (
    "torch::Tensor bmm_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code
bmm_module = load_inline(
    name="bmm_custom",
    cpp_sources=bmm_cpp_source,
    cuda_sources=bmm_source,
    functions=["bmm_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Performs batched matrix multiplication (C = A * B) where A, B, and C have the same batch dimension.
    Optimized with a custom CUDA kernel using tiled shared memory access.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.bmm_func = bmm_module.bmm_cuda
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs batched matrix multiplication using custom CUDA kernel.

        Args:
            A: Input tensor of shape (batch_size, m, k).
            B: Input tensor of shape (batch_size, k, n).

        Returns:
            C: Output tensor of shape (batch_size, m, n).
        """
        return self.bmm_func(A, B)