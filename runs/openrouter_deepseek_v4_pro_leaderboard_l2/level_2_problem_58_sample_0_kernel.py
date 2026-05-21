import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for fused LogSumExp + HardSwish-like + subtract bias + clamp
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_logsumexp_hardswish_sub_clamp_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float bias,
    const int N,
    const int C,
    const int D,
    const int H,
    const int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * D * H * W;
    if (idx >= total) return;

    // Decode linear index to (n, d, h, w)
    int n = idx / (D * H * W);
    int rem = idx % (D * H * W);
    int d = rem / (H * W);
    int rem2 = rem % (H * W);
    int h = rem2 / W;
    int w = rem2 % W;

    // Base pointer to the spatial location (n, 0, d, h, w)
    const float* base = input + n * C * D * H * W + d * H * W + h * W + w;

    // Compute logsumexp over C with numerical stability
    float max_val = -INFINITY;
    for (int c = 0; c < C; ++c) {
        float v = base[c * D * H * W];
        if (v > max_val) max_val = v;
    }

    float sum_exp = 0.0f;
    for (int c = 0; c < C; ++c) {
        sum_exp += expf(base[c * D * H * W] - max_val);
    }

    float logsumexp = max_val + logf(sum_exp);

    // HardSwish-like: x * sigmoid(x + 3) / 6
    float sigmoid_arg = logsumexp + 3.0f;
    float sigmoid_val = 1.0f / (1.0f + expf(-sigmoid_arg));
    float val = logsumexp * sigmoid_val / 6.0f;

    // Subtract bias and clamp
    val = val - bias;
    val = fminf(fmaxf(val, -1.0f), 1.0f);

    output[idx] = val;
}

torch::Tensor fused_logsumexp_hardswish_sub_clamp_cuda(
    torch::Tensor input,
    torch::Tensor bias_tensor)
{
    // Ensure input is contiguous
    input = input.contiguous();

    const int N = input.size(0);
    const int C = input.size(1);
    const int D = input.size(2);
    const int H = input.size(3);
    const int W = input.size(4);

    auto output = torch::empty({N, 1, D, H, W}, input.options());

    float bias = bias_tensor.item<float>();

    const int total = N * D * H * W;
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;

    fused_logsumexp_hardswish_sub_clamp_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        bias,
        N, C, D, H, W);

    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_logsumexp_hardswish_sub_clamp_cuda(
    torch::Tensor input,
    torch::Tensor bias_tensor);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_logsumexp_hardswish_sub_clamp",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_logsumexp_hardswish_sub_clamp_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1))
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_ops.fused_logsumexp_hardswish_sub_clamp_cuda(x, self.bias)
        return x