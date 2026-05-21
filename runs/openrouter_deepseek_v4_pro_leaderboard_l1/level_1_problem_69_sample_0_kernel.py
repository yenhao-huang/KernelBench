```python
import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# CUDA source for transposed 2D convolution (FP32)
conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int N, int C_in, int C_out, int H_in, int W_in,
    int H_out, int W_out,
    int K_h, int K_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int output_padding_h, int output_padding_w
) {
    int idx = blockIdx.x;
    int total_output_spatial = H_out * W_out;
    int total_per_channel = C_out * total_output_spatial;
    int n = idx / total_per_channel;
    int rem = idx % total_per_channel;
    int c_out = rem / total_output_spatial;
    int spatial_idx = rem % total_output_spatial;
    int h_out = spatial_idx / W_out;
    int w_out = spatial_idx % W_out;

    int inner_loop_size = C_in * K_h * K_w;
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    float sum = 0.0f;
    for (int i = tid; i < inner_loop_size; i += blockSize) {
        int c_in = i / (K_h * K_w);
        int rem_k = i % (K_h * K_w);
        int k_h = rem_k / K_w;
        int k_w = rem_k % K_w;

        int h_in = (h_out + padding_h - dilation_h * k_h);
        if (h_in % stride_h != 0) continue;
        h_in /= stride_h;
        int w_in = (w_out + padding_w - dilation_w * k_w);
        if (w_in % stride_w != 0) continue;
        w_in /= stride_w;

        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
            float input_val = input[n * C_in * H_in * W_in + c_in * H_in * W_in + h_in * W_in + w_in];
            // weight shape: (C_in, C_out, K_h, K_w)
            float weight_val = weight[c_in * C_out * K_h * K_w + c_out * K_h * K_w + k_h * K_w + k_w];
            sum += input_val * weight_val;
        }
    }

    sdata[tid] = sum;
    __syncthreads();

    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata