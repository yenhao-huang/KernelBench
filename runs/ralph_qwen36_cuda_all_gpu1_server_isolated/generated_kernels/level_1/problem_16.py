import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication (C = A^T * B)
# We assume A is (K, M) and B is (K, N).
# The operation is torch.matmul(A.T, B), which results in C of shape (M, N).
# C[i, j] = sum_{k=0}^{K-1} A[k, i] * B[k, j]

matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Kernel for Matrix Multiplication: C = A^T * B
// A is (K, M), B is (K, N), C is (M, N)
// Each thread computes one element of C.
__global__ void matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int K, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Index in M dimension
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Index in N dimension

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            // A is stored as (K, M), so A[k, row] is at index k * M + row
            // B is stored as (K, N), so B[k, col] is at index k * N + col
            sum += A[k * M + row] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Validate inputs
    if (!A.is_cuda() || !B.is_cuda()) {
        throw std::runtime_error("Inputs must be CUDA tensors");
    }
    
    int K = A.size(0);
    int M = A.size(1);
    int N = B.size(1);

    // Output tensor shape (M, N)
    auto C = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    const int block_size_x = 32;
    const int block_size_y = 32;
    
    dim3 block(block_size_x, block_size_y);
    dim3 grid((N + block_size_x - 1) / block_size_x, (M + block_size_y - 1) / block_size_y);

    matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), K, M, N);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }

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
    Optimized model that performs a single matrix multiplication (C = A * B)
    using a custom CUDA operator.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_op = matmul_op

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using the custom CUDA kernel.

        Args:
            A: Input tensor of shape (K, M).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        return self.matmul_op.matmul_cuda(A, B)