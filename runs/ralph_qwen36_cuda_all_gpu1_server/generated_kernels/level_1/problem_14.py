import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for upper triangular matrix multiplication
# We optimize by only computing the upper triangular part of the result.
# C[i, j] = sum_k A[i, k] * B[k, j] for i <= j.
# Since A and B are upper triangular:
# A[i, k] is non-zero only if k >= i.
# B[k, j] is non-zero only if j >= k.
# Therefore, the sum is over k from max(i, lower_bound) to min(j, upper_bound).
# Specifically, for a fixed (i, j) with i <= j:
# We need k such that i <= k <= j.
# So C[i, j] = sum_{k=i}^{j} A[i, k] * B[k, j].

upper_triangular_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void upper_triangular_matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
    // Each thread computes one element C[i][j] where i <= j
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Total number of elements in the upper triangle (including diagonal)
    // Sum from i=0 to N-1 of (N - i) = N*(N+1)/2
    int total_elements = N * (N + 1) / 2;
    
    if (idx >= total_elements) return;

    // Map linear index idx to (i, j) coordinates in the upper triangle
    // We can determine i by finding how many elements are in rows 0 to i-1.
    // Row k has N - k elements.
    // Cumulative count before row i: sum_{k=0}^{i-1} (N - k) = i*N - i*(i-1)/2
    // We need to find i such that cumulative_count(i) <= idx < cumulative_count(i+1).
    
    // Approximate i using quadratic formula or binary search. 
    // For simplicity and performance, we can use a simple loop or math.
    // Let's solve for i: i*N - i^2/2 + i/2 = idx => i^2 - (2N+1)i + 2*idx = 0
    // This might be slightly inaccurate due to integer arithmetic, so let's use a safer approach.
    
    // Alternative mapping: 
    // Iterate rows. Row 0 has N elements. Row 1 has N-1...
    // We can compute i by solving the quadratic equation approximately and correcting.
    
    int i = (int)((2.0 * N + 1 - sqrt((double)(2*N+1)*(2*N+1) - 8.0 * idx)) / 2.0);
    // Correct if necessary
    while (i < N && (i * N - i * (i - 1) / 2) > idx) {
        i--;
    }
    while (i + 1 < N && ((i + 1) * N - (i + 1) * i / 2) <= idx) {
        i++;
    }
    
    int start_idx_in_row = i * N - i * (i - 1) / 2;
    int j_offset = idx - start_idx_in_row;
    int j = i + j_offset;

    // Compute dot product for C[i][j]
    float sum = 0.0f;
    // k ranges from i to j because A[i,k] is non-zero only if k>=i and B[k,j] is non-zero only if k<=j
    for (int k = i; k <= j; ++k) {
        sum += A[i * N + k] * B[k * N + j];
    }
    
    C[i * N + j] = sum;
}

torch::Tensor upper_triangular_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    auto N = A.size(0);
    auto out = torch::zeros({N, N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    int total_elements = N * (N + 1) / 2;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    upper_triangular_matmul_kernel<<<num_blocks, block_size>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        out.data_ptr<float>(), 
        N
    );

    return out;
}
"""

upper_triangular_matmul_cpp_source = (
    "torch::Tensor upper_triangular_matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for upper triangular matrix multiplication
upper_triangular_matmul = load_inline(
    name="upper_triangular_matmul",
    cpp_sources=upper_triangular_matmul_cpp_source,
    cuda_sources=upper_triangular_matmul_source,
    functions=["upper_triangular_matmul_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication for upper triangular matrices
    using a custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs optimized matrix multiplication for upper triangular matrices.

        Args:
            A (torch.Tensor): Upper triangular matrix of shape (N, N).
            B (torch.Tensor): Upper triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The product of A and B, also an upper triangular matrix of shape (N, N).
        """
        return upper_triangular_matmul.upper_triangular_matmul_cuda(A, B)