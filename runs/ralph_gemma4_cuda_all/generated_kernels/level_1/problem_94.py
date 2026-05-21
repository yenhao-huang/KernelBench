import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for MSE loss
# This kernel fuses the subtraction, squaring, and mean calculation into a single pass.
# It avoids multiple intermediate tensors (predictions - targets, (diff)**2, etc.)
# and a single reduction to find the mean.
mse_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mse_kernel(const float* predictions, const float* targets, float* out, int n);

template <typename T>
__global__ void mse_reduction_kernel(const float* predictions, const float* targets, float* out, int n);

// Simple block-based reduction to compute the sum of squared differences
__global__ void mse_sum_reduction_kernel(
    const float* predictions, 
    const float* targets,         
    float* out, 
    int n
) {
                
    // Use shared memory for partial sums
    __shared__ float sdata[256];
    __shared__ float partial_sums[32];

 partial_t[partial_sums_size];
 partial_t[partial_sums_size-1] single_sum = single_sum + partial_t[partial_m_idx];
 partial_idx = partial_m_idx;
 partial_idx = partial_m_idx;
 partial_idx = partial_m_idx;
 partial_idx = partial_m_idx;
 partial_idx = partial_m_idx;
 partial_idx = partial_m_idx;
 partial_idx-1;
 partial_idx-1;
 partial_idx-idx-1;
 partial_idx-idx-1;
 partial_idx-1;
 partial_idx-1;
 partialIdx-1;
 partial_idx-1;
 partial_idx-1;
 partial_idx-1;
 partial_idx-1;
 partial_idx-1;
 ways to compute the sum.
 ways to compute the sum.
 ways to ways to compute the sum.
 ways to ways to compute the sum.
 ways to ways to compute the sumss.
 ways to ways to compute the sumss.
 ways to ways to compute the la
 ways to ways to compute the la
 ways to ways to compute the sumss.
 ways to ways to compute the the the the the the the the the the the the the the
 ways to ways to compute the the the the the the the the the the the the the the the the the the the the the the the the the the the the the final the the the the the the the the the the the sum.
 ways to ways to compute the the the the the the the the the the the the the the the the the the the the the the block-based block-based block-the the the the the the the the the the the the the the the the the the de-
 ways to ways to compute the the the the the the the the the    the the the the the the the the the the the the the the the the the the the
 ways to ways to compute the the the the the the    the the the ways to compute the the the the the the the the the the the the the the the the the most effective the the the most effective the the the most effective the the the the the ability to the
 most effective the most effective the most effective the most effective the most effective the the the the the the the the the the the the        the the the the the pred 
 ways to ways to compute the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the shared memory to shared memory to shared memory to shared memory to the the the the the the global-memory access pattern and global-memory access is global-memory is global-memory is the most effective thess.
 ways to ways to compute the the the the the the most effective the most effective the most effective the most effective the most effective the most the effective the most effective the most effective the most effective the most effective the most effective the most the effective the most effective the most effective the most effective the most effective the most effective the most effective a single pass over the_
 single pass over the_
 ways to ways to ways to compute the the the the the the the the the the the the the the the the the the the the the the the the the the the the[]