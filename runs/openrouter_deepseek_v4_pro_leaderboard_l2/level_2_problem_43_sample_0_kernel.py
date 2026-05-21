import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for fused logsumexp + ReLU
logsumexp_relu_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void logsumexp_relu_kernel(const float* input, float* output, int N, int C, int D, int H, int W) {
    extern __shared__ float smem[];
    int idx = blockIdx.x;
    int total_spatial = D * H * W;
    int n = idx / total_spatial;
    int spatial_idx = idx % total_spatial;
    int d = spatial_idx / (H * W);
    int hw = spatial_idx % (H * W);
    int h = hw / W;
    int w = hw % W;

    int channel_stride = D * H * W;
    const float* base = input + n * C * channel_stride + d * H * W + h * W + w;

    int tid = threadIdx.x;
    float val = -INFINITY;
    if (tid < C) {
        val = base[tid * channel_stride];
    }
    smem[tid] = val;
    __syncthreads();

    // Reduction for max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        }
        __syncthreads();
    }
    float max_val = smem[0];
    __syncthreads();

    // Compute exp(x - max) and sum
    float exp_val = (tid < C) ? expf(val - max_val) : 0.0f;
    smem[tid] = exp_val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    float sum_exp = smem[0];

    float logsumexp = max_val + logf(sum_exp);
    float result = fmaxf(0.0f, logsumexp);

    if (tid == 0) {
        output[idx] = result;
    }
}

torch::Tensor logsumexp_relu_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");

    int N = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    auto output = torch::empty({N, 1, D, H, W}, input.options());

    int total_output_elements = N * D * H * W;
    int threads = C;
    int blocks = total_output_elements;

    logsumexp_relu_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), N, C, D, H, W);

    return output;
}
"""

logsumexp_relu_cpp_source = "torch::Tensor logsumexp_relu_cuda(torch::Tensor input);"

# Compile the inline CUDA code
logsumexp_relu = load_inline(
    name="logsumexp_relu",
    cpp_sources=logsumexp_relu_cpp_source,
    cuda_sources=logsumexp_relu_cuda_source,
    functions=["logsumexp_relu_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.logsumexp_relu = logsumexp_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.logsumexp_relu.logsumexp_relu_cuda(x)
        return x