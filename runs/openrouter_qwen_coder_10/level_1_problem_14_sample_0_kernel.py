import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for upper triangular matrix multiplication
triu_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void triu_matmul_kernel(const float* A, const float* B, float* C, int N) {
    // Each thread computes one element C[row, col] of the upper triangular matrix
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= N || col >= N || row > col) {
        // Only compute upper triangular part (row <= col)
        return;
    }
    
    // Compute dot product for C[row, col] = sum(A[row, k] * B[k, col]) for k from row to col
    float sum = 0.0f;
    for (int k = row; k <= col; k++) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

torch::Tensor triu_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);
    
    // Use 2D block of 16x16 threads
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    
    triu_matmul_kernel<<<gridDim, blockDim>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    
    return C;
}
"""

triu_matmul_cpp_source = (
    "torch::Tensor triu_matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for upper triangular matrix multiplication
triu_matmul = load_inline(
    name="triu_matmul",
    cpp_sources=triu_matmul_cpp_source,
    cuda_sources=triu_matmul_source,
    functions=["triu_matmul_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication for upper triangular matrices
    using custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.triu_matmul = triu_matmul
    
    def forward(self, A, B):
        """
        Performs optimized matrix multiplication for upper triangular matrices.

        Args:
            A (torch.Tensor): Upper triangular matrix of shape (N, N).
            B (torch.Tensor): Upper triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The product of A and B, also an upper triangular matrix of shape (N, N).
        """
        return self.triu_matmul.triu_matmul_cuda(A, B)