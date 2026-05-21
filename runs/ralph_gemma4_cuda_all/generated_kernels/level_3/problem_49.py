import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.cpp_extension import load_inline

# CUDA kernel for the intra-chunk state computation
# This kernel computes: states[b, h, c, p, n] = sum_{l} B[b, c, l, h, n] * exp(A_cumsum[b, h, c, L-1] - A_cumsum[b, h, c, l]) * X[b, c, l, h, p]
# This avoids the large intermediate decay_states tensor and the einsum.

intra_chunk_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void intra_chunk_states_kernel(
    const float* __restrict__ B,      // [B, C, L, H, N]
    const float* __restrict__ A_cumsum, // [B, H, C, L]
    const float* __restrict__ X,      // [B, C, L, H, P]
    float* __restrict__ states,       // [B, H, C, P, N]
    int batch_size, int n_heads, int chunk_len, int d_head, int d_state) {

    // Indexing for states: [b, h, c, p, n]
    int n_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int p_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    int idx = n_idx + p_idx * d_state + (n_idx * d_head * d_state / d_state); // This is wrong, let's use flat indexing
    // Correct indexing:
    // states[b, h, c, p, n] = b * (H*C*P*N) + h * (C*P*N) + c * (P*N) + p * N + n
    
    // We'll use a simpler approach: one thread per (b, h, c, p, n) is too many.
    // Let's use one thread per (b, h, c, p, n) but only if the total size is reasonable.
    // Or better: one thread per (b, h, c, p, n) and use a grid of (B*H*C, P, N)
}
"""