import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for multiplying two lower triangular matrices.
# Since A and B are lower triangular, C[i][j] = sum(A[i][k] * B[k][j]) for k <= j and k <= i.
# Also, since we only care about the lower triangle of C, we only compute for j <= i.
# Combining constraints: for a given (i, j) where j <= i, we need k such that k <= j AND k <= i.
# Since j <= i, the condition k <= j implies k <= i. So k ranges from 0 to j.
# This reduces the inner loop from N to j+1 iterations on average (N/2), providing a significant speedup over standard matmul.

lower_triangular_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void lower_triangular_matmul_kernel(const float* A, const float* B, float* C, int N) {
    // Each thread computes one element C[i][j] where j <= i
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < N && j <= i) {
        float sum = 0.0f;
        // k ranges from 0 to j because B[k][j] is non-zero only for k <= j, 
        // and A[i][k] is non-zero only for k <= i. Since j <= i, k <= j is the tighter constraint.
        for (int k = 0; k <= j; ++k) {
            sum += A[i * N + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
    }
}

torch::Tensor lower_triangular_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must have the same size");

    int N = A.size(0);
    auto C = torch::zeros({N, N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    const int block_size_x = 16;
    const int block_size_y = 16;
    
    dim3 block(block_size_x, block_size_y);
    dim3 grid((N + block_size_x - 1) / block_size_x, (N + block_size_y - 1) / block_size_y);

    lower_triangular_matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    return C;
}
"""

lower_triangular_matmul_cpp_source = (
    "torch::Tensor lower_triangular_matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code
lower_triangular_matmul = load_inline(
    name="lower_triangular_matmul",
    cpp_sources=lower_triangular_matmul_cpp_source,
    cuda_sources=lower_triangular_matmul_source,
    functions=["lower_triangular_matmul_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication (C = A * B) where A and B are lower triangular matrices.
    Uses a custom CUDA kernel to exploit the sparsity structure of lower triangular matrices.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication of lower triangular matrices A and B using a custom CUDA kernel.

        Args:
            A (torch.Tensor): Lower triangular matrix of shape (N, N).
            B (torch.Tensor): Lower triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The result of matrix multiplication C of shape (N, N), which is also lower triangular.
        """
        return lower_triangular_matmul.lower_triangular_matmul_cuda(A, B)