import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D tensor-matrix multiplication
matmul_3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_3d_kernel(const float* A, const float* B, float* C, int N, int M, int K, int L) {
    // Thread indices
    int n = blockIdx.z;
    int m = blockIdx.y;
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (n >= N || m >= M || l >= L) return;
    
    // Compute one element of the output: C[n, m, l]
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[n * (M * K) + m * K + k] * B[k * L + l];
    }
    C[n * (M * L) + m * L + l] = sum;
}

torch::Tensor matmul_3d_cuda(torch::Tensor A, torch::Tensor B) {
    // Get dimensions
    int N = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int L = B.size(1);
    
    // Create output tensor
    auto C = torch::empty({N, M, L}, A.options());
    
    // Configure kernel launch parameters
    const int block_size = 256;
    dim3 block_dims(block_size);
    dim3 grid_dims((L + block_size - 1) / block_size, M, N);
    
    // Launch kernel
    matmul_3d_kernel<<<grid_dims, block_dims>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        C.data_ptr<float>(), 
        N, M, K, L
    );
    
    return C;
}
"""

matmul_3d_cpp_source = (
    "torch::Tensor matmul_3d_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for 3D tensor-matrix multiplication
matmul_3d = load_inline(
    name="matmul_3d",
    cpp_sources=matmul_3d_cpp_source,
    cuda_sources=matmul_3d_source,
    functions=["matmul_3d_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized 3D tensor-matrix multiplication using custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_3d = matmul_3d
    
    def forward(self, A, B):
        """
        Performs 3D tensor-matrix multiplication using optimized CUDA kernel.

        Args:
            A (torch.Tensor): Input 3D tensor of shape (N, M, K).
            B (torch.Tensor): Input matrix of shape (K, L).

        Returns:
            torch.Tensor: Output tensor of shape (N, M, L), resulting from the multiplication of A and B along the last dimension of A.
        """
        return self.matmul_3d.matmul_3d_cuda(A, B)

N = 16
M = 1024
K = 2048
L = 768

def get_inputs():
    A = torch.rand(N, M, K)
    B = torch.rand(K, L)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed