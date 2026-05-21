import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for triangular matrix multiplication.
# Since A and B are lower triangular, C = A * B is also lower triangular.
# C[i, j] = sum_{k=0}^{min(i, j)} A[i, k] * B[k, j]
# This kernel optimizes by only computing the lower triangular part and 
# limiting the inner loop to the non-zero range.
triangular_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void triangular_matmul_kernel(const float* __restrict__ A, 
                                         const float* __restrict__ B, 
                                         float* __restrict__ out, 
                                         int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        if (row < col) {
            // Upper triangular part is zero
            out[row * N + col] = 0.0f;
        } else {
            // Lower triangular part: k must be <= row AND k must be <= col
            // Since we are in the 'else' block, row >= col, so k <= col.
            float sum = 0.0f;
            for (int k = 0; k <= col; ++k) {
                sum += A[row * N + k] * B[k * N + col];
            }
            out[row * N + col] = sum;
        }
    }
}

torch::Tensor triangular_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    auto N = A.size(0);
    auto out = torch::zeros_like(A);

    dim3 block_size(16, 16);
    dim3 grid_size((N + block_size.x - 1) / block_size.x, (N + block_size.y - 1) / block_size.y);

    triangular_matmul_kernel<<<grid_size, block_size>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        out.data_ptr<float>(), 
        N
    );

    return out;
}
"""

triangular_matmul_cpp_source = """
torch::Tensor triangular_matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

# Compile the inline CUDA code
triangular_matmul_lib = load_inline(
    name="triangular_matmul_lib",
    cpp_sources=triangular_matmul_cpp_source,
    cuda_sources=triangular_matmul_source,
    functions=["triangular_matmul_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication (C = A * B) 
    where A and B are lower triangular matrices using a custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.triangular_matmul = triangular_matmul_lib.triangular_matmul_cuda
    
    def forward(self, A, B):
        """
        Performs matrix multiplication of lower triangular matrices A and B.

        Args:
            A (torch.Tensor): Lower triangular matrix of shape (N, N).
            B (torch.Tensor): Lower triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The result of matrix multiplication C of shape (N, N).
        """
        # Ensure inputs are contiguous and on CUDA for the custom kernel
        if not A.is_cuda:
            A = A.cuda()
        if not B.is_cuda:
            B = B.cuda()
            
        return self.triangular_matmul(A.contiguous(), B.contiguous())