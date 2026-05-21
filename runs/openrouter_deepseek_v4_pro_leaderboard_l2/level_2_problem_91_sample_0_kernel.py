import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for the fused kernel
fused_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_softmax_bias_scale_sigmoid_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float scaling_factor,
    float* __restrict__ output,
    int N, int C, int H, int W)
{
    // Each block handles one spatial location (n, h, w)
    int n = blockIdx.x / (H * W);
    int residual = blockIdx.x % (H * W);
    int h = residual / W;
    int w = residual % W;

    extern __shared__ float shared_data[]; // used for max and sum reductions
    float* shared_max = shared_data;       // first half for max reduction
    float* shared_exp = shared_data;       // reuse for exp/sum reduction

    int tid = threadIdx.x;
    float my_val = 0.0f;

    // Load input value for this channel at the given spatial location
    if (tid < C) {
        int idx = ((n * C + tid) * H + h) * W + w;
        my_val = input[idx];
    }

    // Max reduction
    if (tid < C) {
        shared_max[tid] = my_val;
    }
    __syncthreads();

    for (int stride = C / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float a = shared_max[tid];
            float b = shared_max[tid + stride];
            shared_max[tid] = (a > b) ? a : b;
        }
        __syncthreads();
    }
    float max_val = shared_max[0];
    __syncthreads();

    // Compute exp(x - max) and store in shared memory for sum reduction
    if (tid < C) {
        float exp_val = expf(my_val - max_val);
        shared_exp[tid] = exp_val;
    }
    __syncthreads();

    // Sum reduction
    for (int stride = C / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_exp[tid] += shared_exp[tid + stride];
        }
        __syncthreads();
    }
    float sum = shared_exp[0];
    __syncthreads();

    // Compute final output: softmax -> add bias -> scale -> sigmoid
    if (tid < C) {
        float exp_val = expf(my_val - max_val);
        float softmax_val = exp_val / sum;
        float biased = softmax_val + bias[tid];
        float scaled = biased * scaling_factor;
        float sigmoid_val = 1.0f / (1.0f + expf(-scaled));
        int out_idx = ((n * C + tid) * H + h) * W + w;
        output[out_idx] = sigmoid_val;
    }
}

torch::Tensor fused_softmax_bias_scale_sigmoid_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    float scaling_factor)
{
    // Assume input is 4D: (N, C, H, W)
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    auto output = torch::zeros_like(input);

    const int threads = C;  // one thread per channel
    const int blocks = N * H * W;  // one block per spatial location
    const int shared_mem_size = C * sizeof(float);

    fused_softmax_bias_scale_sigmoid_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        scaling_factor,
        output.data_ptr<float>(),
        N, C, H, W);

    return output;
}
"""

fused_cpp_source = (
    "torch::Tensor fused_softmax_bias_scale_sigmoid_cuda("
    "torch::Tensor input, torch::Tensor bias, float scaling_factor);"
)

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_softmax_bias_scale_sigmoid",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_cuda_source,
    functions=["fused_softmax_bias_scale_sigmoid_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_op.fused_softmax_bias_scale_sigmoid_cuda(x, self.bias, self.scaling_factor)
        return x