import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Tiled matrix multiplication kernel
__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Tile size
    const int TILE_M = 16;
    const int TILE_N = 16;
    const int TILE_K = 16;
    
    // Shared memory for tiles
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_N][TILE_K];
    
    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Global row and column indices
    int row = blockIdx.y * TILE_M + ty;
    int col = blockIdx.x * TILE_N + tx;
    
    // Accumulator for the result
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_K - 1) / TILE_K; ++t) {
        // Load tile from A to shared memory
        if (row < M && t * TILE_K + tx < K) {
            As[ty][tx] = A[row * K + t * TILE_K + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Load tile from B to shared memory
        if (col < N && t * TILE_K + ty < K) {
            Bs[tx][ty] = B[(t * TILE_K + ty) * N + col];
        } else {
            Bs[tx][ty] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial product for this tile
        for (int k = 0; k < TILE_K; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    // Check dimensions
    TORCH_CHECK(A.size(1) == B.size(0), "Matrix dimensions must match for multiplication");
    
    // Create output tensor
    auto C = torch::empty({M, N}, A.options());
    
    // Configure kernel launch parameters
    const int TILE_M = 16;
    const int TILE_N = 16;
    
    dim3 threads(TILE_N, TILE_M);
    dim3 blocks((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    
    matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return C;
}
"""

matmul_cpp_source = (
    "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for matrix multiplication
matmul_module = load_inline(
    name="matmul_cuda",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication using custom CUDA kernel
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_module = matmul_module
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication of A and B using optimized CUDA kernel.

        Args:
            A: Input tensor with shape (M, K).
            B: Input tensor with shape (K, N).

        Returns:
            C: Output tensor with shape (M, N).
        """
        return self.matmul_module.matmul_cuda(A, B)