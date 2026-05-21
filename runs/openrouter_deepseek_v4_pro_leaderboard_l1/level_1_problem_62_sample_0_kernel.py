```python
import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Custom CUDA im2col kernel source
im2col_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void im2col_kernel(
    const float* input,
    float* columns,
    int N, int C, int H, int W,
    int groups, int C_per_group,
    int kH, int kW,
    int stride, int pad_h, int pad_w, int dilation,
    int H_out, int W_out,
    int total_elements
) {
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (col_idx >= total_elements) return;

    int out_spatial_size = H_out * W_out;
    int filter_entry_size = C_per_group * kH * kW;

    int spatial_idx = col_idx % out_spatial_size;
    int filter_idx = (col_idx / out_spatial_size) % filter_entry_size;
    int g = (col_idx / (out_spatial_size * filter_entry_size)) % groups;
    int n = col_idx / (out_spatial_size * filter_entry_size * groups);

    int c_in = filter_idx % C_per_group;
    filter_idx /= C_per_group;
    int kw = filter_idx % kW;
    int kh = filter_idx / kW;

    int c_global = g * C_per_group + c_in;

    int w_out = spatial_idx % W_out;
    int h_out = spatial_idx / W_out;

    int h_in = h_out * stride - pad_h + kh * dilation;
    int w_in = w_out * stride - pad_w + kw * dilation;

    float val = 0.0f;
    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
        val = input[((n * C + c_global) * H + h_in) * W + w_in];
    }
    columns[col_idx] = val;
}

torch::Tensor im2col_cuda(
    torch::Tensor input,
    int kH, int kW,
    int stride, int pad_h, int pad_w,
    int dilation, int groups
) {
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    int C_per_group = C / groups;

    int H_out = (H + 2 * pad_h - dilation * (kH - 1) - 1) / stride + 1;
    int W_out = (W + 2 * pad_w - dilation * (kW - 1) - 1) / stride + 1;

    int out_spatial_size = H_out * W_out;
    int filter_entry_size = C_per_group * kH * kW;
    int total_elements = N * groups * filter_entry_size * out_spatial_size;

    auto columns = torch::empty({total_elements}, input.options());

    const int block_size = 256;