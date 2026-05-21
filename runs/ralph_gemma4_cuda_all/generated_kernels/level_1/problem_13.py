import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# The problem asks to optimize a matrix multiplication C = A * B where A and B are symmetric.
# While A and B are symmetric, C is not necessarily symmetric.
# For N=4096, the most efficient way to perform this on a GPU is to use highly optimized 
# libraries like cuBLAS, which PyTorch's torch.matmul already uses.
# However, to fulfill the requirement of a "custom CUDA operator", we implement a 
# tiled matrix multiplication kernel. For large N, we use shared memory tiling 
# to improve data reuse and reduce global memory bandwidth pressure.

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void tiled_matmul_kernel(const float* __restrict__ A, 
                                   const float* __restrict__ B, 
                                   float* __restrict__ C, 
                                   int N) {
    // Tiling parameters
    const int TILE_SIZE = 32;
    
    // Shared memory for tiles of A and B
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int m = 0; m < (N + TILE_SIZE - 1) / TILE_SIZE; ++m) {
        // Load tiles into shared memory
        if (row < N && (m * TILE_SIZE + threadIdx.x) < N) {
            s_A[threadIdx.y][threadIdx.x] = A[row * N + (m * TILE_SIZE + threadIdx.x)];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && (m * TILE_SIZE + threadIdx.y) < N) {
            s_B[threadIdx.y][threadIdx.x] = B[(m * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial product using the tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor tiled_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);

    dim3 dimBlock(32, 32);
    dim3 dimGrid((N + 31) / 32, (N + 31) / 32);

    tiled_matmul_kernel<<<dimGrid, dimBlock>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        C.data_ptr<float>(), 
        N
    );

    return C;
}
"""

cpp_source = "torch::Tensor tiled_matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the custom CUDA operator
tiled_matmul_lib = load_inline(
    name="tiled_matmul_lib",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["tiled_matmul_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Optimized model using a custom tiled CUDA kernel for matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tiled_matmul = tiled_matmul_lib.tiled_matmul_cuda

    def forward(self, A, B):
        """
        Performs matrix multiplication of two symmetric matrices using a custom CUDA kernel.

        Args:
            A (torch.Tensor): Input matrix A, shape (N, N), symmetric, on CUDA.
            B (torch.Tensor): Input matrix B, shape (N, N), symmetric, on CUDA.

        Returns:
            torch.Tensor: Output matrix C, shape (N, N), on CUDA.
        """
        # Ensure inputs are contiguous and on CUDA for the custom kernel
        A = A.contiguous().cuda()
        B = B.contiguous().cuda()
        return self.tiled_matmul(A, B)