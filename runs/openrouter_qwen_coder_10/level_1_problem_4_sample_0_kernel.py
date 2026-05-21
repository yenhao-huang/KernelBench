import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix-vector multiplication
gemv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Optimized GEMV kernel using shared memory for B vector to reduce global memory accesses
__global__ void gemv_shared_kernel(const float* A, const float* B, float* C, int M, int K) {
    extern __shared__ float shared_B[];
    
    // Load B vector into shared memory
    int tid = threadIdx.x;
    for (int i = tid; i < K; i += blockDim.x) {
        shared_B[i] = B[i];
    }
    __syncthreads();
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M) {
        float sum = 0.0f;
        // Process K elements with loop unrolling
        int k = 0;
        for (; k + 7 < K; k += 8) {
            sum += A[row * K + k] * shared_B[k] +
                   A[row * K + k + 1] * shared_B[k + 1] +
                   A[row * K + k + 2] * shared_B[k + 2] +
                   A[row * K + k + 3] * shared_B[k + 3] +
                   A[row * K + k + 4] * shared_B[k + 4] +
                   A[row * K + k + 5] * shared_B[k + 5] +
                   A[row * K + k + 6] * shared_B[k + 6] +
                   A[row * K + k + 7] * shared_B[k + 7];
        }
        // Handle remaining elements
        for (; k < K; k++) {
            sum += A[row * K + k] * shared_B[k];
        }
        C[row] = sum;
    }
}

torch::Tensor gemv_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    
    auto C = torch::zeros({M, 1}, A.options());
    
    const int block_size = 256;
    const int num_blocks = (M + block_size - 1) / block_size;
    
    size_t shared_mem_size = K * sizeof(float);
    
    gemv_shared_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        C.data_ptr<float>(), 
        M, K
    );
    
    return C;
}

torch::Tensor gemv(torch::Tensor A, torch::Tensor B) {
    return gemv_cuda(A, B);
}
"""

gemv_cpp_source = """
torch::Tensor gemv(torch::Tensor A, torch::Tensor B);
"""

# Compile the inline CUDA code for matrix-vector multiplication
gemv_module = load_inline(
    name="gemv",
    cpp_sources=gemv_cpp_source,
    cuda_sources=gemv_source,
    functions=["gemv"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix-vector multiplication (C = A * B) using custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gemv = gemv_module
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix-vector multiplication using optimized CUDA kernel.

        Args:
            A: Input matrix of shape (M, K).
            B: Input vector of shape (K, 1).

        Returns:
            Output vector of shape (M, 1).
        """
        return self.gemv.gemv(A, B)