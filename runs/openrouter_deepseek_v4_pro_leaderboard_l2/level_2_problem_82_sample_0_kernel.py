import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused tanh, scaling, bias addition, and max pooling
fused_tanh_scale_bias_maxpool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_tanh_scale_bias_maxpool_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ bias,
    float scaling_factor,
    int N, int C, int H, int W,
    int pool_size, int H_out, int W_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (idx >= total) return;

    // Compute output indices
    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c = (idx / (W_out * H_out)) % C;
    int n = idx / (W_out * H_out * C);

    float max_val = -1e30f;
    float bias_val = bias[c];

    int h_start = h_out * pool_size;
    int w_start = w_out * pool_size;
    int h_end = min(h_start + pool_size, H);
    int w_end = min(w_start + pool_size, W);

    for (int h = h_start; h < h_end; ++h) {
        for (int w = w_start; w < w_end; ++w) {
            float val = input[n * C * H * W + c * H * W + h * W + w];
            val = tanhf(val) * scaling_factor + bias_val;
            if (val > max_val) max_val = val;
        }
    }
    output[idx] = max_val;
}

torch::Tensor fused_tanh_scale_bias_maxpool_cuda(
    torch::Tensor input,
    float scaling_factor,
    torch::Tensor bias,
    int pool_size)
{
    // Get input dimensions
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    // Compute output dimensions (assuming stride = pool_size, padding = 0)
    int H_out = H / pool_size;
    int W_out = W / pool_size;

    // Allocate output tensor
    auto output = torch::zeros({N, C, H_out, W_out}, input.options());

    // Launch kernel
    int total_threads = N * C * H_out * W_out;
    const int block_size = 256;
    const int num_blocks = (total_threads + block_size - 1) / block_size;

    fused_tanh_scale_bias_maxpool_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        bias.data_ptr<float>(),
        scaling_factor,
        N, C, H, W,
        pool_size, H_out, W_out
    );

    return output;
}
"""

fused_tanh_scale_bias_maxpool_cpp_source = (
    "torch::Tensor fused_tanh_scale_bias_maxpool_cuda("
    "torch::Tensor input, float scaling_factor, torch::Tensor bias, int pool_size);"
)

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_tanh_scale_bias_maxpool",
    cpp_sources=fused_tanh_scale_bias_maxpool_cpp_source,
    cuda_sources=fused_tanh_scale_bias_maxpool_source,
    functions=["fused_tanh_scale_bias_maxpool_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.pool_kernel_size = pool_kernel_size
        self.fused_op = fused_op

    def forward(self, x):
        # Convolution (kept as standard PyTorch op)
        x = self.conv(x)
        # Fused tanh, scaling, bias addition, and max pooling
        x = self.fused_op.fused_tanh_scale_bias_maxpool_cuda(
            x, self.scaling_factor, self.bias, self.pool_kernel_size
        )
        return x