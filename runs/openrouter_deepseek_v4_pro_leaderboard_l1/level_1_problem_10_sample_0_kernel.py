import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for batched matrix multiplication
batched_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void batched_matmul_kernel(const float* A, const float* B, float* C,
                                      int R, int K, int L) {
    // Block indices
    int bx = blockIdx.x;  // column tile in L
    int by = blockIdx.y;  // row tile in R

    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Base row and column for this block
    int base_row = by * blockDim.y;
    int base_col = bx * blockDim.x;

    // Global row and column for this thread
    int row = base_row + ty;
    int col = base_col + tx;

    // Shared memory tiles
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];

    float sum = 0.0f;

    // Loop over K dimension in tiles of 16
    for (int k_tile = 0; k_tile < K; k_tile += 16) {
        // Load A tile: A[row, k_tile + tx]
        if (row < R && (k_tile + tx) < K) {
            As[ty][tx] = A[row * K + (k_tile + tx)];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load B tile: B[k_tile + ty, base_col + tx]
        if ((k_tile + ty) < K && (base_col + tx) < L) {
            Bs[ty][tx] = B[(k_tile + ty) * L + (base_col + tx)];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // Write result
    if (row < R && col < L) {
        C[row * L + col] = sum;
    }
}

torch::Tensor batched_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // A: (N, M, K), B: (K, L)
    const auto N = A.size(0);
    const auto M = A.size(1);
    const auto K = A.size(2);
    const auto L = B.size(1);

    // Flatten A to (N*M, K)
    auto A_flat = A.reshape({N * M, K}).contiguous();
    auto R = N * M;

    // Allocate output (N*M, L)
    auto C_flat = torch::zeros({R, L}, A.options());

    // Launch kernel
    const dim3 block(16, 16);
    const dim3 grid((L + 15) / 16, (R + 15) / 16);

    batched_matmul_kernel<<<grid, block>>>(
        A_flat.data_ptr<float>(),
        B.data_ptr<float>(),
        C_flat.data_ptr<float>(),
        R, K, L);

    // Reshape back to (N, M, L)
    return C_flat.reshape({N, M, L});
}
"""

batched_matmul_cpp_source = """
torch::Tensor batched_matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

# Compile the inline CUDA code
batched_matmul = load_inline(
    name="batched_matmul",
    cpp_sources=batched_matmul_cpp_source,
    cuda_sources=batched_matmul_source,
    functions=["batched_matmul_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.batched_matmul = batched_matmul

    def forward(self, A, B):
        return self.batched_matmul.batched_matmul_cuda(A, B)