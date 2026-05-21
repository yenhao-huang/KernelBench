import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for matrix-vector multiplication
matvec_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void matvec_kernel(const float* A, const float* B, float* C, int M, int K) {
    int row = blockIdx.x;
    if (row >= M) return;
    __shared__ float sB[BLOCK_SIZE];
    int tid = threadIdx.x;
    float sum = 0.0f;
    const float* A_row = A + row * K;
    for (int k = 0; k < K; k += BLOCK_SIZE) {
        int idx = k + tid;
        if (idx < K) {
            sB[tid] = B[idx];
        } else {
            sB[tid] = 0.0f;
        }
        __syncthreads();
        if (idx < K) {
            sum += A_row[idx] * sB[tid];
        }
        __syncthreads();
    }
    // Block reduction
    sB[tid] = sum;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sB[tid] += sB[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        C[row] = sB[0];
    }
}

torch::Tensor matvec_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");
    auto M = A.size(0);
    auto K = A.size(1);
    // Handle B shape (K, 1) by squeezing to 1D
    auto B_squeezed = B.squeeze();
    TORCH_CHECK(B_squeezed.dim() == 1 && B_squeezed.size(0) == K, "B must have shape (K,) or (K,1)");
    auto C = torch::empty({M}, A.options());
    const dim3 grid(M);
    const dim3 block(BLOCK_SIZE);
    matvec_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B_squeezed.data_ptr<float>(),
        C.data_ptr<float>(),
        M,
        K
    );
    return C;
}
"""

matvec_cpp_source = "torch::Tensor matvec_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the custom operator
matvec_op = load_inline(
    name="matvec_cuda",
    cpp_sources=matvec_cpp_source,
    cuda_sources=matvec_cuda_source,
    functions=["matvec_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Simple model that performs matrix-vector multiplication (C = A * B)
    using a custom CUDA kernel optimized for FP32.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matvec = matvec_op

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix-vector multiplication.

        Args:
            A: Input matrix of shape (M, K).
            B: Input vector of shape (K, 1).

        Returns:
            Output vector of shape (M, 1).
        """
        # Call custom CUDA kernel that returns shape (M,)
        out = self.matvec.matvec_cuda(A.contiguous(), B.contiguous())
        # Reshape to (M, 1) to match original interface
        return out.unsqueeze(1)