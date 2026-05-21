import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for symmetric matrix multiplication
# Since A and B are symmetric, we can optimize by only computing the upper/lower triangular parts
# and exploiting symmetry. However, for simplicity and to ensure correctness, we implement
# a standard matrix multiplication kernel that can be optimized by the compiler.
# We use a tiled approach for better memory coalescing and shared memory usage.

symmetric_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

__global__ void symmetric_matmul_kernel(const float* A, const float* B, float* C, int N) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    float sum = 0.0f;

    for (int k = 0; k < N; k += BLOCK_SIZE) {
        // Load tiles into shared memory
        if (row < N && (k + tx) < N)
            As[ty][tx] = A[row * N + (k + tx)];
        else
            As[ty][tx] = 0.0f;

        if (col < N && (k + ty) < N)
            Bs[ty][tx] = B[(k + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        // Compute partial dot product
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            sum += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor symmetric_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    symmetric_matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    return C;
}
"""

symmetric_matmul_cpp_source = (
    "torch::Tensor symmetric_matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code
symmetric_matmul = load_inline(
    name="symmetric_matmul",
    cpp_sources=symmetric_matmul_cpp_source,
    cuda_sources=symmetric_matmul_source,
    functions=["symmetric_matmul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.symmetric_matmul = symmetric_matmul

    def forward(self, A, B):
        return self.symmetric_matmul.symmetric_matmul_cuda(A, B)