import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix-scalar multiplication
matrix_scalar_mult_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matrix_scalar_mult_kernel(const float* A, float* C, float s, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = M * N;
    
    if (idx < total_elements) {
        C[idx] = A[idx] * s;
    }
}

torch::Tensor matrix_scalar_mult_cuda(torch::Tensor A, float s) {
    auto M = A.size(0);
    auto N = A.size(1);
    auto out = torch::empty_like(A);

    const int block_size = 256;
    int total_elements = M * N;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    matrix_scalar_mult_kernel<<<num_blocks, block_size>>>(A.data_ptr<float>(), out.data_ptr<float>(), s, M, N);

    return out;
}
"""

matrix_scalar_mult_cpp_source = (
    "torch::Tensor matrix_scalar_mult_cuda(torch::Tensor A, float s);"
)

# Compile the inline CUDA code for matrix-scalar multiplication
matrix_scalar_mult = load_inline(
    name="matrix_scalar_mult",
    cpp_sources=matrix_scalar_mult_cpp_source,
    cuda_sources=matrix_scalar_mult_source,
    functions=["matrix_scalar_mult_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix-scalar multiplication (C = A * s)
    using a custom CUDA operator.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matrix_scalar_mult = matrix_scalar_mult

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        """
        Performs matrix-scalar multiplication using custom CUDA kernel.

        Args:
            A: Input matrix of shape (M, N)
            s: Scalar value

        Returns:
            C: Resulting matrix of shape (M, N)
        """
        return self.matrix_scalar_mult.matrix_scalar_mult_cuda(A, s)