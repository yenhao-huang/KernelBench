```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA implementation of fused masked cumulative sum along the last dimension
masked_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 1024

__global__ void segment_scan_kernel(const float* __restrict__ x, const bool* __restrict__ mask,
                                    float* __restrict__ out, float* __restrict__ segment_sums,
                                    int B, int N, int num_segments, int seg_len) {
    int seg_idx = blockIdx.x;
    int row = blockIdx.y;
    int start = seg_idx * seg_len;
    int end = min(start + seg_len, N);
    int seg_size = end - start;
    
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = start + tid;
    float val = 0.0f;
    if (tid < seg_size) {
        val = mask[row * N + idx] ? x[row * N + idx] : 0.0f;
    }
    sdata[tid] = val;
    __syncthreads();
    
    // Inclusive scan within the block
    for (int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
        float temp;
        if (tid >= stride) {
            temp = sdata[tid] + sdata[tid - stride];
        }
        __syncthreads();
        if (tid >= stride) {
            sdata[tid] = temp;
        }
        __syncthreads();
    }
    
    if (tid < seg_size) {
        out[row * N + start + tid] = sdata[tid];
    }
    if (tid == seg_size - 1) {
        segment_sums[row * num_segments + seg_idx] = sdata[seg_size - 1];
    }
}

__global__ void compute_offsets_kernel(const float* segment_sums, float* segment_offsets,
                                        int num_segments, int B, int block_sz) {
    int row = blockIdx.x;
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    if (tid < num_segments) {
        sdata[tid] = segment_sums[row * num_segments + tid];
    } else {
        sdata[tid] = 0.0f;
    }
    __syncthreads();
    
    // Inclusive scan over segment sums
    for (int stride = 1; stride < block_sz; stride *= 2) {
        float temp = 0.0f;
        if (tid >= stride) {
            temp = sdata[tid] + sdata[tid - stride];
        }
        __syncthreads();
        if (tid >= stride) {
            sdata[tid] = temp;
        }
        __syncthreads();
    }
    
    if (tid < num_segments) {
        float off = (tid == 0) ? 0.0f : sdata[tid - 1];
        segment_offsets[row * num_segments + tid] = off;
    }
}

__global__ void add_offsets_kernel(float* out, const float* segment_offsets,
                                    int B,