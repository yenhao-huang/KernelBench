```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for A^T * B^T with tiled shared memory
gemm_custom_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_M 32
#define TILE_N 32
#define TILE_K 32

__global__ void gemm_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = bx