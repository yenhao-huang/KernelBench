import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for fused min(dim=1) + sum(dim=2) + GELU + bias
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ float gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f; // sqrt(2/pi)
    const float coeff = 0.044715f;
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__global__ void fused_min_sum_gelu_add_kernel(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int B, int C, int H, int W) {

    int w = blockIdx.x;
    int b = blockIdx.y;
    int tid = threadIdx.x;

    // Base offset for this (b, w) pair: x[b, 0, 0, w]
    int base = b * (C * H * W) + w;

    extern __shared__ float sdata[];

    float sum = 0.0f;

    for (int h = 0; h < H; ++h) {
        // Each thread loads one channel value for this h
        int c = tid;
        float val = (c < C) ? x[base + c * H * W + h * W] : INFINITY;
        sdata[tid] = val;
        __syncthreads();

        // Parallel reduction to find min over channels
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                sdata[tid] = fminf(sdata[tid], sdata[tid + stride]);
            }
            __syncthreads();
        }

        // Thread 0 accumulates the min for this h
        if (tid == 0) {
            sum += sdata[0];
        }
        __syncthreads();
    }

    // Only thread 0 writes the final result
    if (tid == 0) {
        float result = gelu(sum) + bias[0];
        out[b * W + w] = result;
    }
}

torch::Tensor fused_min_sum_gelu_add_cuda(
    torch::Tensor x,
    torch::Tensor bias) {

    // x shape: (B, C, H, W)
    int B = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    auto out = torch::empty({B, 1, 1, W}, x.options());

    const int threads = 128; // C is 128, one thread per channel
    dim3 blocks(W, B);
    size_t shared_mem = threads * sizeof(float);

    fused_min_sum_gelu_add_kernel<<<blocks, threads, shared_mem>>>(
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        B, C, H, W);

    return out;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_min_sum_gelu_add_cuda(torch::Tensor x, torch::Tensor bias);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_min_sum_gelu_add",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_min_sum_gelu_add_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_op.fused_min_sum_gelu_add_cuda(x, self.bias)
        return x