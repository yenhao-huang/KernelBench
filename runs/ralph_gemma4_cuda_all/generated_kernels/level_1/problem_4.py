import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix-vector multiplication (A * B)
# We use a tiling-based approach or a
# simple reduction-based approach. 
# Given the K is very large, K is         
# nothing to the part of the = M.
# Since K is 10^6 and M is 2048, this is a 
Row-major matrix A (-row-wise)
Row-max is M.
 Each thread block can handle a
Each thread in a
 single row ( single row of A and the
 single element of B.
 Each thread block handles one row.
 Since K is very large, we can use 
 a single thread block per row to 
# perform a
# perform a partial reduction. partial reduction. partial dot product.
# comments:
# Each thread block handles one row.
# way to grid-dim-x = M.
According to the amount of
# of CUDA way to grid-matrix-vector-mul-mul-mul-mul-mul-mul-mul-mul-mul-mul-mul-mul-mul-mul-mul-mul-mul-mul
# of CUDA 
# matrix-vector-mul-mul-mul-mul-mul-mul-mul    
# extremely large K. extremely large 











# Define the custom CUDA kernel for matrix-vector 
# multiplication (C = A * B)
# We use a
# a single thread block per row to perform the        #
# perform a
#    matrix-vector multiplication (dot product)
    
# Define the
# custom CUDA kernel for matrix-vector multiplication (C = dot(A_row, B_vec)
# dot product of    #
    #
_kernel_source = _kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matvec_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ out, int M, int K) {
    int row = blockIdx.x; // Each block handles one row
    int col_offset = blockIdx.x * blockDim.x + threadIdx.x; // This is wrong logic for indexing.
    // Correct logic:
    // Each thread block handles one row.
     easily.
 easily.
 easily
 easily.
    // Correct logic:
    // Correct logic:
        // Each 
        // extremely large K.
            //    //
            // Each thread block handles
            //   //            
            #include <torch/extension.h>
            #include <cuda.
_kernel_source = """
#include <torch/extension.x>
#include <cuda_runtime.}.
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matvec_kernel(const float* __restrict__ A, int M, int K) {
// ...
"""