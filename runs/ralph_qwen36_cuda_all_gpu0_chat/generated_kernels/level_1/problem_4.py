import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix-vector multiplication (GEMV)
# This implementation uses shared memory tiling to optimize global memory access patterns.
gemv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Kernel for Matrix-Vector Multiplication: C = A * B
// A is (M, K), B is (K, 1), C is (M, 1)
// We process rows of A in blocks. Each block handles a chunk of rows.
// To optimize memory access to B, we load chunks of B into shared memory.

__global__ void gemv_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int K) {
    // Each thread block handles a tile of rows from A.
    // Let's say each block processes 'BLOCK_M' rows.
    // We will use dynamic shared memory or fixed size depending on constraints.
    // Given K is large (1048576), we cannot load all B into shared memory.
    // Strategy: Load a tile of B into shared memory, process multiple rows from A against this tile.
    
    // Define block dimensions
    const int BLOCK_M = 32; // Number of rows per block
    const int TILE_K = 1024; // Number of elements of B to load into shared memory at a time
    
    extern __shared__ float s_B[];

    int row_start = blockIdx.y * BLOCK_M;
    
    // Each thread in the block handles one row within the block's tile
    int local_row_idx = threadIdx.x; 
    
    if (row_start + local_row_idx >= M) {
        return;
    }

    float sum = 0.0f;
    
    // Iterate over tiles of K
    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        int current_k_end = min(k_tile + TILE_K, K);
        int tile_size = current_k_end - k_tile;
        
        // Load tile of B into shared memory
        // We use a simple loop or unrolled load. Since tile_size is constant-ish, we can optimize.
        // However, for simplicity and correctness with variable last tile:
        if (threadIdx.x < tile_size) {
            s_B[threadIdx.x] = B[k_tile + threadIdx.x];
        } else {
            s_B[threadIdx.x] = 0.0f;
        }
        __syncthreads();

        // Compute dot product for the specific row
        int global_row = row_start + local_row_idx;
        
        // We need to access A[global_row][k_tile ... current_k_end)
        // The pointer to the start of this row's segment in A
        const float* a_ptr = &A[global_row * K + k_tile];
        
        #pragma unroll
        for (int i = 0; i < TILE_K; ++i) {
            if (k_tile + i < K) {
                sum += a_ptr[i] * s_B[i];
            }
        }
        
        __syncthreads();
    }

    // Write result to C
    C[row_start + local_row_idx] = sum;
}

torch::Tensor gemv_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2 && B.size(1) == 1, "B must be 2D with shape (K, 1)");

    int M = A.size(0);
    int K = A.size(1);
    
    // Validate dimensions match
    TORCH_CHECK(K == B.size(0), "Dimension mismatch between A and B");

    auto C = torch::zeros({M, 1}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    const int BLOCK_M = 32;
    const int TILE_K = 1024;
    
    // Number of threads per block is equal to BLOCK_M (each thread handles one row)
    const int threads_per_block = BLOCK_M;
    
    // Calculate number of blocks needed for rows
    const int num_blocks_y = (M + BLOCK_M - 1) / BLOCK_M;
    
    // Shared memory size: TILE_K floats
    const size_t shared_mem_size = TILE_K * sizeof(float);

    dim3 block(threads_per_block);
    dim3 grid(1, num_blocks_y);

    gemv_kernel<<<grid, block, shared_mem_size>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K);

    return C;
}
"""

gemv_cpp_source = (
    "torch::Tensor gemv_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for GEMV
gemv = load_inline(
    name="gemv",
    cpp_sources=gemv_cpp_source,
    cuda_sources=gemv_source,
    functions=["gemv_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix-vector multiplication using a custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gemv = gemv

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix-vector multiplication using the custom CUDA kernel.

        Args:
            A: Input matrix of shape (M, K).
            B: Input vector of shape (K, 1).

        Returns:
            Output vector of shape (M, 1).
        """
        return self.gemv.gemv_cuda(A, B)