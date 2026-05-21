import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for diagonal matrix-vector multiplication (broadcasted)
# This performs C[i, j] = A[i] * B[i, j]
diagonal_broadcast_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void diagonal_broadcast_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ out, int N, int M) {
    // Each thread handles one element of the output matrix C[N][M]
    // Row index i corresponds to the index in A
    // Column index j corresponds to the column in B
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Total number of elements is N * M
    if (idx < N * M) {
        int i = idx / M; // Row index
        int j = idx % M; // Column index
        
        out[idx] = A[i] * B[idx];
    }
}

torch::Tensor diagonal_broadcast_cuda(torch::Tensor A, torch::Tensor B) {
    auto N = A.size(0);
    auto M = B.size(1);
    
    TORCH_CHECK(A.numel() == N, "A must be 1D tensor of size N");
    TORCH_CHECK(B.sizes()[0] == N && B.sizes()[1] == M, "B must be 2D tensor of size (N, M)");

    auto out = torch::empty_like(B);

    const int block_size = 256;
    const int total_elements = N * M;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    diagonal_broadcast_kernel<<<num_blocks, block_size>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        out.data_ptr<float>(), 
        N, 
        M
    );

    return out;
}
"""

diagonal_broadcast_cpp_source = (
    "torch::Tensor diagonal_broadcast_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code
diagonal_broadcast = load_inline(
    name="diagonal_broadcast",
    cpp_sources=diagonal_broadcast_cpp_source,
    cuda_sources=diagonal_broadcast_source,
    functions=["diagonal_broadcast_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication of a diagonal matrix with another matrix.
    C = diag(A) * B
    Uses a custom CUDA kernel for better performance on large tensors.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.diagonal_broadcast = diagonal_broadcast

    def forward(self, A, B):
        """
        Performs the matrix multiplication using custom CUDA operator.

        Args:
            A (torch.Tensor): A 1D tensor representing the diagonal of the diagonal matrix. Shape: (N,).
            B (torch.Tensor): A 2D tensor representing the second matrix. Shape: (N, M).

        Returns:
            torch.Tensor: The result of the matrix multiplication. Shape: (N, M).
        """
        return self.diagonal_broadcast.diagonal_broadcast_cuda(A, B)