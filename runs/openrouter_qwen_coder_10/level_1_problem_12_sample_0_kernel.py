import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for diagonal matrix multiplication
diag_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void diag_matmul_kernel(const float* A, const float* B, float* C, int N, int M) {
    // Each thread computes one element of the result matrix C
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < M) {
        // C[row, col] = A[row] * B[row, col]
        C[row * M + col] = A[row] * B[row * M + col];
    }
}

torch::Tensor diag_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // A is 1D tensor of size N
    // B is 2D tensor of size (N, M)
    // Output C is 2D tensor of size (N, M)
    
    int N = A.size(0);
    int M = B.size(1);
    
    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    auto C = torch::empty({N, M}, options);
    
    // Configure CUDA kernel launch parameters
    const int block_x = 32;
    const int block_y = 32;
    dim3 threads(block_x, block_y);
    dim3 blocks((M + block_x - 1) / block_x, (N + block_y - 1) / block_y);
    
    diag_matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N, M);
    
    return C;
}
"""

diag_matmul_cpp_source = (
    "torch::Tensor diag_matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for diagonal matrix multiplication
diag_matmul = load_inline(
    name="diag_matmul",
    cpp_sources=diag_matmul_cpp_source,
    cuda_sources=diag_matmul_source,
    functions=["diag_matmul_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication of a diagonal matrix with another matrix.
    C = diag(A) * B
    Uses custom CUDA kernel for efficient computation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.diag_matmul = diag_matmul
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication.

        Args:
            A (torch.Tensor): A 1D tensor representing the diagonal of the diagonal matrix. Shape: (N,).
            B (torch.Tensor): A 2D tensor representing the second matrix. Shape: (N, M).

        Returns:
            torch.Tensor: The result of the matrix multiplication. Shape: (N, M).
        """
        return self.diag_matmul.diag_matmul_cuda(A, B)