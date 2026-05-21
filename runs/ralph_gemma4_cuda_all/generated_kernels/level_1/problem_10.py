import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# The operation is torch.matmul(A, B) where A is (N, M, K) and B is (K, L).
# This is equivalent to performing (N*M) independent matrix-vector multiplications
# or treating the first two dimensions as a batch.
# Since B is shared across all (N, M) slices, we can optimize this by 
# using a single large batch matrix multiplication (BMM) or a GEMM.
# In CUDA, the most efficient way to handle this is to reshape A to (N*M, K)
# and perform a standard GEMM: (N*M, K) x (K, L) -> (N*M, L).
# PyTorch's torch.matmul already does a good job, but we can implement a 
# custom kernel that avoids the overhead of reshaping and uses a single 
# kernel launch for the entire batch.

tensor_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// We use a tiled approach or simply rely on the fact that for large K and L,
// a standard GEMM is best. However, to provide a custom CUDA implementation 
// that is robust, we implement a kernel that computes the batch GEMM.
// For high performance, one would typically use cuBLAS, but here we 
// implement a kernel that demonstrates the custom operator capability.

__global__ void batch_matmul_kernel(const float* __restrict__ A, 
                                   const float* __restrict__ B, 
                                   float* __restrict__ C, 
                                   int NM, int K, int L) {
    // Each thread handles one element of the output C[nm][l]
    int nm = blockIdx.y * blockDim.y + threadIdx.y;
    int l = blockIdx.x * blockDim.x + threadIdx.x;

    if (nm < NM && l < L) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[nm * K + k] * B[k * L + l];
        }
        C[nm * L + l] = sum;
    }
}

torch::Tensor batch_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    auto N = A.size(0);
    auto M = A.size(1);
    auto K = A.size(2);
    auto L = B.size(1);

    auto C = torch::empty({N, M, L}, A.options());

    int NM = N * M;
    
    // Using a 2D grid for the output dimensions (NM, L)
    // We use a block size of 16x16 or 32x32
    dim3 block(32, 32);
    dim3 grid((L + block.x - 1) / block.x, (NM + block.y - 1) / block.y);

    batch_matmul_kernel<<<grid, block>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        C.data_ptr<float>(), 
        NM, K, L
    );

    return C;
}
"""

tensor_matmul_cpp_source = """
torch::Tensor batch_matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

# Compile the inline CUDA code
batch_matmul_lib = load_inline(
    name="batch_matmul_lib",
    cpp_sources=tensor_matmul_cpp_source,
    cuda_sources=tensor_matmul_source,
    functions=["batch_matmul_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized 3D tensor-matrix multiplication using a custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.batch_matmul_cuda = batch_matmul_lib.batch_matmul_cuda
    
    def forward(self, A, B):
        """
        Args:
            A (torch.Tensor): Input 3D tensor of shape (N, M, K).
            B (torch.Tensor): Input matrix of shape (K, L).

        Returns:
            torch.Tensor: Output tensor of shape (N, M, L).
        """
        # Ensure inputs are contiguous for the custom kernel
        if not A.is_contiguous():
            A = A.contiguous()
        if not B.is_contiguous():
            B = B.contiguous()
            
        return self.batch_matmul_cuda(A, B)