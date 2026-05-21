import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for batched matrix multiplication
gemm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Constants for tiling
constexpr int TILE_M = 32;
constexpr int TILE_N = 32;
constexpr int TILE_K = 32;

__global__ void batched_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size,
    int m,
    int n,
    int k
) {
    // Each block computes a TILE_M x TILE_N tile of the output matrix
    int batch_idx = blockIdx.z;
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    
    // Shared memory for tiles
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_N][TILE_K];
    
    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Compute global row and column indices
    int row = block_row * TILE_M + ty;
    int col = block_col * TILE_N + tx;
    
    // Accumulator for the result
    float acc = 0.0f;
    
    // Loop over tiles of K dimension
    int num_tiles = (k + TILE_K - 1) / TILE_K;
    
    for (int t = 0; t < num_tiles; ++t) {
        int k_start = t * TILE_K;
        
        // Load tile from A into shared memory
        if (row < m && k_start + tx < k) {
            As[ty][tx] = A[batch_idx * m * k + row * k + k_start + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Load tile from B into shared memory
        if (col < n && k_start + ty < k) {
            Bs[tx][ty] = B[batch_idx * k * n + (k_start + ty) * n + col];
        } else {
            Bs[tx][ty] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product for this tile
        #pragma unroll
        for (int i = 0; i < TILE_K; ++i) {
            acc += As[ty][i] * Bs[tx][i];
        }
        
        __syncthreads();
    }
    
    // Write result to global memory
    if (row < m && col < n) {
        C[batch_idx * m * n + row * n + col] = acc;
    }
}

torch::Tensor batched_gemm_cuda(torch::Tensor A, torch::Tensor B) {
    // Get dimensions
    int batch_size = A.size(0);
    int m = A.size(1);
    int k = A.size(2);
    int n = B.size(2);
    
    // Create output tensor
    auto C = torch::empty({batch_size, m, n}, A.options());
    
    // Configure grid and block dimensions
    dim3 block_dim(TILE_N, TILE_M, 1);
    dim3 grid_dim(
        (n + TILE_N - 1) / TILE_N,
        (m + TILE_M - 1) / TILE_M,
        batch_size
    );
    
    // Launch kernel
    batched_gemm_kernel<<<grid_dim, block_dim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size,
        m,
        n,
        k
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\\n", cudaGetErrorString(err));
    }
    
    return C;
}
"""

gemm_cpp_source = (
    "torch::Tensor batched_gemm_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for batched matrix multiplication
batched_gemm = load_inline(
    name="batched_gemm",
    cpp_sources=gemm_cpp_source,
    cuda_sources=gemm_source,
    functions=["batched_gemm_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom CUDA kernel for batched matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.batched_gemm = batched_gemm
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs batched matrix multiplication using optimized CUDA kernel.

        Args:
            A: Input tensor of shape (batch_size, m, k).
            B: Input tensor of shape (batch_size, k, n).

        Returns:
            C: Output tensor of shape (batch_size, m, n).
        """
        return self.batched_gemm.batched_gemm_cuda(A, B)