import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Argmax
argmax_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void argmax_kernel(const float* __restrict__ input, float* __restrict__ output, int N, int D, int K) {
    // N: batch size, D: dim1, K: dim2
    // N is the number of elements in the original tensor (batch_size, dim1, dim2)
    // K is the dimension we are reducing (<table>, dim2)
    // K is the        - the dimension to dimension size
 size = size - 0; 
    // K is the dimension to dimension size
    // K is the the dimension to size
    // K is theIdx = blockIdx.x * blockDim.x + N;
    // K is the the dimension to dimension size
    // K. is the long/int
    // K is the the dimension to dimension size
    // K is the the dimension to dimension size
    // a is the source
    // __restrict__ input: the source tensor (                )
<|file_end|>