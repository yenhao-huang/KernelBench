import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source code for fused clamp + softmax + scale kernel
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void clamp_softmax_scale_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ scale,
    int N,
    int C,
    float clamp_min,
    float clamp_max)
{
    int b = blockIdx.x;          // index = batch * C + channel
    int c = b % C;
    float sc = scale[c];

    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Step 1: find maximum of clamped values
    float local_max = -INFINITY;
    for (int i = tid; i < N; i += stride) {
        float val = x[b * N + i];
        val = fminf(fmaxf(val, clamp_min), clamp_max);
        local_max = fmaxf(local_max, val);
    }
    shared[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }
    float M = shared[0];
    __syncthreads();

    // Step 2: compute sum of exp(x - M)
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += stride) {
        float val = x[b * N + i];
        val = fminf(fmaxf(val, clamp_min), clamp_max);
        local_sum += expf(val - M);
    }
    shared[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    float S = shared[0];
    __syncthreads();

    // Step 3: compute output = softmax * scale
    for (int i = tid; i < N; i += stride) {
        float val = x[b * N + i];
        val = fminf(fmaxf(val, clamp_min), clamp_max);
        float softmax_val = expf(val - M) / S;
        y[b * N + i] = softmax_val * sc;
    }
}

torch::Tensor fused_clamp_softmax_scale_cuda(
    torch::Tensor x,
    float clamp_min,
    float clamp_max,
    torch::Tensor scale)
{
    auto B = x.size(0);
    auto C = x.size(1);
    auto D = x.size(2);
    auto H = x.size(3);
    auto W = x.size(4);
    int N = D * H * W;

    // Flatten spatial dimensions
    auto x_flat = x.reshape({B * C, N}).contiguous();
    auto scale_flat = scale.reshape({C}).contiguous();
    auto y = torch::zeros_like(x_flat);

    const int block_size = 256;
    dim3 grid(B * C);
    size_t shared_mem_size = block_size * sizeof(float);

    clamp_softmax_scale_kernel<<<grid, block_size, shared_mem_size>>>(
        x_flat.data_ptr<float>(),
        y.data_ptr<float>(),
        scale_flat.data_ptr<float>(),
        N,
        C,
        clamp_min,
        clamp_max
    );

    y = y.reshape({B, C, D, H, W});
    return y;
}
"""

fused_op_cpp_source = """
torch::Tensor fused_clamp_softmax_scale_cuda(
    torch::Tensor x,
    float clamp_min,
    float clamp_max,
    torch::Tensor scale);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_clamp_softmax_scale",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_clamp_softmax_scale_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernel fusing clamp, softmax, and scale multiplication.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))
        self.fused_op = fused_op

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv_transpose(x)
        x = self.fused_op.fused_clamp_softmax_scale_cuda(x, self.clamp_min, self.clamp_max, self.scale)
        return x