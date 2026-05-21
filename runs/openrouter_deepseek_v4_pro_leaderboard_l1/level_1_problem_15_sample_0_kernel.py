import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for lower triangular matrix multiplication
tril_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE 32

__global__ void tril_matmul_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    if (row >= N || col >= N) return;

    float acc = 0.0f;

    // Iterate over tiles along the K dimension
    for (int kTile = 0; kTile < N; kTile += TILE) {
        // Shared memory for tiles of A and B
        __shared__ float As[TILE][TILE];
        __shared__ float Bs[TILE][TILE];

        // Load tile from A (row, kTile+tx)
        if (row < N && (kTile + threadIdx.x) < N) {
            As[threadIdx.y][threadIdx.x] = A[row * N + (kTile + threadIdx.x)];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile from B (kTile+ty, col)
        if (col < N && (kTile + threadIdx.y) < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(kTile + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the result only for the lower triangular part (row >= col)
    if (row >= col) {
        C[row * N + col] = acc;
    }
}

torch::Tensor tril_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);

    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    tril_matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    return C;
}
"""

tril_matmul_cpp_source = "torch::Tensor tril_matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code
tril_matmul = load_inline(
    name="tril_matmul",
    cpp_sources=tril_matmul_cpp_source,
    cuda_sources=tril_matmul_source,
    functions=["tril_matmul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication of lower triangular matrices
    using a custom CUDA kernel that computes only the lower triangle of the result.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tril_matmul = tril_matmul

    def forward(self, A, B):
        """
        Args:
            A (torch.Tensor): Lower triangular matrix of shape (N, N) on GPU.
            B (torch.Tensor): Lower triangular matrix of shape (N, N) on GPU.

        Returns:
            torch.Tensor: The result of lower triangular matrix multiplication C of shape (N, N).
        """
        return self.tril_matmul.tril_matmul_cuda(A, B)


M = 4096

def get_inputs():
    A = torch.rand(M, M).cuda()
    B = torch.rand(M, M).cuda()
    A = torch.tril(A)
    B = torch.tril(B)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed