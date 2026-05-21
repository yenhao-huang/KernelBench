import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D tensor-matrix multiplication (Batched MatMul)
# This kernel performs C[i, j, k] = sum_l A[i, j, l] * B[l, k]
# Optimized for FP32 precision.
batched_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel for batched matrix multiplication: C = A @ B
// A is (N, M, K), B is (K, L), C is (N, M, L)
__global__ void batched_matmul_kernel(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ C, 
    int N, int M, int K, int L) 
{
    // Each thread computes one element of the output matrix for a specific batch and row
    int n = blockIdx.z;       // Batch index: 0 to N-1
    int m = blockIdx.y;       // Row index in A (and C): 0 to M-1
    int l = threadIdx.x;      // Column index in B (and C): 0 to L-1
    
    if (n >= N || m >= M || l >= L) return;

    float sum = 0.0f;
    
    // Pointer arithmetic for A and B
    // A[n, m, :] is at A + n*(M*K) + m*K
    // B[:, l] is at B + l (since B is contiguous in row-major order, column l starts at index l)
    const float* a_row = A + n * M * K + m * K;
    const float* b_col = B + l; 
    
    // Loop over the reduction dimension K
    for (int k = 0; k < K; ++k) {
        sum += a_row[k] * b_col[k * L];
    }
    
    C[n * M * L + m * L + l] = sum;
}

torch::Tensor batched_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 3, "A must be a 3D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    
    int N = A.size(0);
    int M = A.size(1);
    int K_A = A.size(2);
    int K_B = B.size(0);
    int L = B.size(1);
    
    TORCH_CHECK(K_A == K_B, "Inner dimensions of A and B must match");
    
    auto C = torch::zeros({N, M, L}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    const int block_size = 256; // Number of threads per block (one per column L)
    dim3 grid(N, M, 1);         // Grid dimensions: N batches, M rows
    
    batched_matmul_kernel<<<grid, block_size>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N, M, K_A, L);
    
    return C;
}
"""

batched_matmul_cpp_source = (
    "torch::Tensor batched_matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for batched matrix multiplication
batched_matmul = load_inline(
    name="batched_matmul",
    cpp_sources=batched_matmul_cpp_source,
    cuda_sources=batched_matmul_source,
    functions=["batched_matmul_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Performs 3D tensor-matrix multiplication using a custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.batched_matmul = batched_matmul

    def forward(self, A, B):
        """
        Performs 3D tensor-matrix multiplication.

        Args:
            A (torch.Tensor): Input 3D tensor of shape (N, M, K).
            B (torch.Tensor): Input matrix of shape (K, L).

        Returns:
            torch.Tensor: Output tensor of shape (N, M, L), resulting from the multiplication of A and B along the last dimension of A.
        """
        return self.batched_matmul.batched_matmul_cuda(A, B)