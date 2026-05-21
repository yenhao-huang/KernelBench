import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for a specialized GEMM (M, K, N) where K is small.
# This kernel is optimized for FP32.
# It uses a tiling approach where each thread block handles a small tile of the output matrix C.
# Since K is small, we can load the entire K dimension into shared memory or registers.
# For this specific case (M >> K and N is small), we optimize for memory access patterns.

skinny_gemm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void skinny_gemm_kernel(const float* __restrict__ A, 
                                  const float* __restrict__ B, 
                                  float* __restrict__ C, 
                                  int M, int K, int N) {
    // Each thread handles one element of the output matrix C (M, N)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            // A is (M, K), B is (K, N)
            // A[row, k] = A[row * K + k]
            // B[k, col] = B[k * N + col]
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor skinny_gemm_cuda(torch::Tensor A, torch::Tensor B) {
    auto M = A.size(0);
    auto K = A.size(1);
    auto N = B.size(1);
    
    auto C = torch::empty({M, N}, A.options());

    // Define block and grid dimensions
    // Using 16x16 or 32x32 tiles for the output matrix
    dim3 block_size(32, 32);
    dim3 grid_size((N + block_size.x - 1) / block_size.x, 
                   (M + block_size.y - 1) / block_size.y);

    skinny_gemm_kernel<<<grid_size, block_size>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        C.data_ptr<float>(), 
        M, K, N
    );

    return C;
}
"""

skinny_gemm_cpp_source = """
torch::Tensor skinny_gemm_cuda(torch::Tensor A, torch::Tensor B);
"""

# Compile the inline CUDA code
skinny_gemm_lib = load_inline(
    name="skinny_gemm_lib",
    cpp_sources=skinny_gemm_cpp_source,
    cuda_sources=skinny_gemm_source,
    functions=["skinny_gemm_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel for skinny matrix multiplication.
    Designed for cases where K is small relative to M and N.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.skinny_gemm = skinny_gemm_lib

    def forward(self, A, B):
        """
        Performs the matrix multiplication using the custom CUDA kernel.

        Args:
            A (torch.Tensor): Input matrix of shape (M, K).
            B (torch.Tensor): Input matrix of shape (K, N).

        Returns:
            torch.Tensor: Output matrix of shape (M, N).
        """
        # Ensure inputs are contiguous and on CUDA for the custom kernel
        if not A.is_cuda:
            A = A.cuda()
        if not B.is_cuda:
            B = B.cuda()
            
        # The custom kernel expects row-major contiguous tensors
        A = A.contiguous()
        B = B.contiguous()

        return self.skinny_gemm.skinny_gemm_cuda(A, B)