import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for LogSoftmax.
# We use a two-pass approach (online max and then compute log-sum-exp) 
# or a single pass with reduction. For simplicity and stability in a single kernel, 
# we implement a kernel where each block handles one row to compute max and sum.
# Given the large dimension (393216), we use a block-per-row strategy with 
# shared memory reduction to find max and sum.

log_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void log_softmax_kernel(const float* __restrict__ input, float* __restrict__ output, int rows, int cols) {
    // Each block handles one row. 
    // Since cols is very large (393216), we use a grid of blocks where each block 
    // processes a chunk of the row, or more simply, we use a single block per row 
    // if it fits, but 393216 is too large for one block's shared memory.
    // Instead, we use a standard reduction pattern.
    
    // For very large cols, we'll use a multi-pass approach or a highly parallel reduction.
    // Here, we implement a kernel where each thread processes elements and we use 
    // atomic operations or a reduction tree.
    
    // To ensure stability and performance for dim=393216:
    // 1. Find max of the row.
    // 2. Find sum(exp(x - max)).
    // 3. output = x - max - log(sum).
    
    // However, to keep the kernel efficient and avoid excessive global memory access,
    // we'll use a simple approach: each thread handles one element, but we need 
    // to reduce. For simplicity in this inline example, we'll use a 
    // block-based reduction for each row.
    
    int row = blockIdx.y;
    int tid = threadIdx.x;
    int total_threads = blockDim.x;
    
    const float* row_ptr = input + row * cols;
    float* out_ptr = output + row * cols;

    // Step 1: Find Max
    float local_max = -FLT_MAX;
    for (int i = tid; i < cols; i += total_threads) {
        local_max = fmaxf(local_max, row_ptr[i]);
    }

    // Reduction for max (using shared memory)
    extern __shared__ float sdata[];
    sdata[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    float row_max = sdata[0];

    // Step 2: Find Sum of Exps
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += total_threads) {
        local_sum += expf(row_ptr[i] - row_max);
    }

    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = sdata[tid] + sdata[tid + s];
        }
        __syncthreads();
    }
    float row_sum_exp = sdata[0];

    // Step 3: Compute LogSoftmax
    float log_sum_exp = logf(row_sum_exp);
    for (int i = tid; i < cols; i += total_threads) {
        out_ptr[i] = row_ptr[i] - row_max - log_sum_exp;
    }
}

torch::Tensor log_softmax_cuda(torch::Tensor input, int dim) {
    // This implementation assumes dim=1 (the second dimension)
    int rows = input.size(0);
    int cols = input.size(1);
    auto output = torch::empty_like(input);

    // We use a 2D grid: y-dimension is rows, x-dimension is 1.
    // Each block handles one row.
    dim3 grid(1, rows);
    dim3 block(256); // Adjust based on hardware
    size_t shared_mem = 256 * sizeof(float);

    log_softmax_kernel<<<grid, block, shared_mem>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        rows, 
        cols
    );

    return output;
}
"""

log_softmax_cpp_source = """
torch::Tensor log_softmax_cuda(torch::Tensor input, int dim);
"""

# Compile the inline CUDA code
log_softmax_lib = load_inline(
    name="log_softmax_lib",
    cpp_sources=log_softmax_cpp_source,
    cuda_sources=log_softmax_source,
    functions=["log_softmax_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a LogSoftmax activation using a custom CUDA kernel.
    """
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.log_softmax_cuda = log_softmax_lib.log_softmax_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies LogSoftmax activation to the input tensor using custom CUDA kernel.
        Note: This optimized version is specifically tuned for dim=1.
        """
        if x.is_cuda and x.dim() == 2 and self.dim == 1:
            return self.log_softmax_cuda(x, self.dim)
        else:
            return torch.log_softmax(x, dim=self.dim)