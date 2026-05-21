import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication optimized for small K
# We use a tiled approach to maximize register usage and minimize global memory traffic.
# Since K is very small (64), we can unroll loops or use shared memory effectively.
# However, for such small K, a simple block-based approach with careful tiling is often best.
# We will implement a standard GEMM kernel that handles the dimensions M, N, K.

matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Block size for threads in x and y directions
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define TILE_K 4 // Unroll factor for K dimension to improve instruction level parallelism

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Each block computes a TILE_SIZE_X x TILE_SIZE_Y tile of the output matrix C
    __shared__ float As[BLOCK_SIZE_Y][TILE_K]; // Shared memory for A tiles
    __shared__ float Bs[TILE_K][BLOCK_SIZE_X]; // Shared memory for B tiles

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Base indices for the current tile in A and B
    int base_a_row = by * BLOCK_SIZE_Y + ty;
    int base_b_col = bx * BLOCK_SIZE_X + tx;

    float sum = 0.0f;

    // Loop over tiles in K dimension
    for (int k = 0; k < K; k += TILE_K) {
        // Load tile from A into shared memory
        // We assume M and N are large enough that boundary checks are needed if not aligned, 
        // but for simplicity and speed with typical shapes, we check bounds.
        int a_idx = base_a_row * K + k + tx;
        int b_idx = (k + ty) * N + base_b_col;

        // Load A element
        if (base_a_row < M && (k + tx) < K) {
            As[ty][tx] = A[a_idx];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load B element
        if ((k + ty) < K && base_b_col < N) {
            Bs[ty][tx] = B[b_idx];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product for this tile
        #pragma unroll
        for (int i = 0; i < TILE_K; ++i) {
            sum += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }

    // Write result to global memory
    int c_idx = base_a_row * N + base_b_col;
    if (base_a_row < M && base_b_col < N) {
        C[c_idx] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    auto M = A.size(0);
    auto K = A.size(1);
    auto N = B.size(1);

    auto out = torch::zeros({M, N}, A.options());

    const int block_x = BLOCK_SIZE_X;
    const int block_y = BLOCK_SIZE_Y;
    
    // Calculate grid dimensions
    int grid_x = (N + block_x - 1) / block_x;
    int grid_y = (M + block_y - 1) / block_y;

    dim3 block(block_x, block_y);
    dim3 grid(grid_x, grid_y);

    matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), out.data_ptr<float>(), M, N, K);

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
    extra_ldflags=["-lcudart"]
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