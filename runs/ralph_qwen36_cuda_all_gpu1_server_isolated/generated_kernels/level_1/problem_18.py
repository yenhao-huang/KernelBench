import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication (C = A^T * B^T)
# This is equivalent to computing (B * A)^T, but we will implement it directly 
# as C[i][j] = sum_k A[k][i] * B[k][j] to match the shape logic of matmul(A.T, B.T).
# A is (K, M), B is (N, K).
# A.T is (M, K), B.T is (K, N).
# Result C is (M, N).
# C[m][n] = sum_{k=0}^{K-1} A[k][m] * B[n][k]

matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    // Each thread computes one element of the output matrix C[M][N]
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m < M && n < N) {
        float sum = 0.0f;
        // C[m][n] = sum_{k} A[k][m] * B[n][k]
        // Note: In the original torch.matmul(A.T, B.T), 
        // A is (K, M), so A.T is (M, K). Element at (m, k) is A[k][m].
        // B is (N, K), so B.T is (K, N). Element at (k, n) is B[n][k].
        for (int k = 0; k < K; ++k) {
            sum += A[k * M + m] * B[n * K + k];
        }
        C[m * N + n] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // A shape: (K, M)
    // B shape: (N, K)
    // Output shape: (M, N)
    
    int K_val = A.size(0);
    int M_val = A.size(1);
    int N_val = B.size(0);

    auto C = torch::zeros({M_val, N_val}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    const int block_size_x = 32;
    const int block_size_y = 32;
    
    dim3 block(block_size_x, block_size_y);
    dim3 grid((N_val + block_size_x - 1) / block_size_x, 
              (M_val + block_size_y - 1) / block_size_y);

    matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M_val, N_val, K_val);

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
        Performs matrix multiplication using custom CUDA kernel.
        
        Args:
            A: Input tensor of shape (K, M).
            B: Input tensor of shape (N, K).
            
        Returns:
            Output tensor of shape (M, N), equivalent to torch.matmul(A.T, B.T).
        """
        return self.matmul_op.matmul_cuda(A, B)