import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source code for a coalesced argmin kernel along dim=1 of a 3D tensor
cuda_src = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void argmin_dim1_kernel(const float* input, int64_t* output,
                                   int N, int M, int K, int blocks_per_n,
                                   int block_dim_x) {
    int n = blockIdx.x / blocks_per_n;
    int k_block = blockIdx.x % blocks_per_n;
    int k = k_block * blockDim.x + threadIdx.x;
    if (n >= N || k >= K) return;

    // pointer to the first element (m=0) for this (n,k) slice
    int base_idx = n * M * K + k;
    float min_val = input[base_idx];
    int min_idx = 0;

    for (int m = 1; m < M; ++m) {
        float val = input[base_idx + m * K];
        if (val < min_val) {
            min_val = val;
            min_idx = m;
        }
    }
    output[n * K + k] = (int64_t)min_idx;
}

torch::Tensor argmin_dim1_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "x must be 3D");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");

    x = x.contiguous();
    int64_t N = x.size(0);
    int64_t M = x.size(1);
    int64_t K = x.size(2);

    auto output = torch::empty({N, K}, torch::dtype(torch::kInt64).device(x.device()));

    const int block_dim_x = 256;
    int blocks_per_n = (K + block_dim_x - 1) / block_dim_x;
    int total_blocks = N * blocks_per_n;
    dim3 grid(total_blocks);
    dim3 block(block_dim_x);

    argmin_dim1_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        output.data_ptr<int64_t>(),
        N, M, K, blocks_per_n, block_dim_x
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel failed: ", cudaGetErrorString(err));
    }

    return output;
}
"""

cpp_src = "torch::Tensor argmin_dim1_cuda(torch::Tensor x);"

# Compile the inline CUDA code
argmin_custom = load_inline(
    name="argmin_custom",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["argmin_dim1_cuda"],
    verbose=False,
    extra_cflags=[],
    extra_ldflags=[],
)


class ModelNew(nn.Module):
    """
    Optimized version using a custom CUDA kernel for argmin along dim=1
    on 3D float32 tensors. Falls back to torch.argmin for other cases.
    """
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.argmin_custom = argmin_custom

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use the coalesced kernel for the most common case (dim=1, 3D fp32).
        if self.dim == 1 and x.ndim == 3 and x.dtype == torch.float32:
            return self.argmin_custom.argmin_dim1_cuda(x)
        else:
            return torch.argmin(x, dim=self.dim)