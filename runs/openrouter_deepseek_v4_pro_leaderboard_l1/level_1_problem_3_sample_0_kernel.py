import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for batched matrix multiplication
bmm_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define TILE_SIZE 16

__global__ void bmm_kernel(const float* A, const float* B, float* C,
                           int batch_size, int m, int k, int n) {
    // Block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global row and column indices
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    // Batch index
    int batch = bz;

    // Shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    // Loop over tiles of k dimension
    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load A tile into shared memory
        if (row < m && (t * TILE_SIZE + tx) < k) {
            As[ty][tx] = A[batch * m * k + row * k + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load B tile into shared memory
        if ((t * TILE_SIZE + ty) < k && col < n) {
            Bs[ty][tx] = B[batch * k * n + (t * TILE_SIZE + ty) * n + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }

    // Write result
    if (row < m && col < n) {
        C[batch * m * n + row * n + col] = sum;
    }
}

torch::Tensor bmm_cuda(torch::Tensor A, torch::Tensor B) {
    int batch_size = A.size(0);
    int m = A.size(1);
    int k = A.size(2);
    int n = B.size(2);

    auto C = torch::zeros({batch_size, m, n}, A.options());

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (m + BLOCK_SIZE - 1) / BLOCK_SIZE,
              batch_size);

    bmm_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
                                batch_size, m, k, n);

    return C;
}
"""

bmm_cpp_source = "torch::Tensor bmm_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code
bmm_op = load_inline(
    name="bmm_op",
    cpp_sources=bmm_cpp_source,
    cuda_sources=bmm_cuda_source,
    functions=["bmm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.bmm_op = bmm_op

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.bmm_op.bmm_cuda(A, B)