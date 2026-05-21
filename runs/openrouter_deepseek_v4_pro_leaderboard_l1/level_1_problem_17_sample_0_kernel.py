import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for tiled matrix multiplication C = A * B^T
matmul_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // A: M x K, B: N x K, C: M x N
    // Each block computes a TILE_SIZE x TILE_SIZE tile of C
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    for (int k = 0; k < K; k += TILE_SIZE) {
        // Load tile from A into shared memory
        if (row < M && (k + threadIdx.x) < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + (k + threadIdx.x)];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile from B into shared memory (transposed access pattern)
        if ((blockIdx.x * TILE_SIZE + threadIdx.y) < N && (k + threadIdx.x) < K)
            Bs[threadIdx.x][threadIdx.y] = B[(blockIdx.x * TILE_SIZE + threadIdx.y) * K + (k + threadIdx.x)];
        else
            Bs[threadIdx.x][threadIdx.y] = 0.0f;

        __syncthreads();

        // Compute partial dot products
        for (int t = 0; t < TILE_SIZE; ++t) {
            sum += As[threadIdx.y][t] * Bs[t][threadIdx.x];
        }
        __syncthreads();
    }

    // Write result to global memory
    if (row < M && col < N)
        C[row * N + col] = sum;
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // A: (M, K), B: (N, K) -> C: (M, N) = A * B^T
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);  // B is N x K

    auto C = torch::zeros({M, N}, A.options());

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

    return C;
}
"""

matmul_cpp_source = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code
matmul_op = load_inline(
    name="matmul_cuda",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_cuda_source,
    functions=["matmul_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_cuda = matmul_op

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_cuda.matmul_cuda(A, B)