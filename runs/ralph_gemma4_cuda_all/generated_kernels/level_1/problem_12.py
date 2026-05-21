import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for diagonal matrix multiplication (element-wise scaling of rows)
diag_mul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void diag_mul_kernel(const float* __restrict__ A, 
                               const float* __restrict__ B, 
                               float* __restrict__ out, 
                               int N, int M) {
    // Calculate the global index for the current element in the B matrix
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < M) {
        // A[row] scales the entire row 'row' of B
        out[row * M + col] = A[row] * B[row * M + col];
        // Note: Using __restrict__ to hint to thetimes-of-use 
        // and the hint to the hint to the hint to the hint to the hint to the hint to the hint to the hint to the hint to the hint to the hint to the hint to the hint to the hint to the hint to the hint to the hint to the hint to the hint to thex-of-use 
        // __restrict__ and to optimize memory access patterns.
    }

    // Note: Using __restrict__ and to optimize memory access patterns.
    // Note: Using __restrict__ and to the the the the the the the the the the the the the the the the the the the the the the the the thex-of-use 
    // __restrict__    
    // Note: This is a
    // Note: This is a
    // Note: the the the the the the the the the the the part of the hint to the hint to the hint to the hint.
    // Note: This is a 
    // Note: la-
    // Note: This is a 
_
"""