import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source code for the fused kernel
fused_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ subtract,
    float* __restrict__ out,
    int batch_size,
    int in_features,
    int out_features,
    bool has_bias)
{
    extern __shared__ float shared[];
    float* s_x = shared;
    float* s_sum = &shared[in_features];

    int sample_idx = blockIdx.x;
    if (sample_idx >= batch_size) return;

    // Load input x into shared memory
    const float* x_sample = x + sample_idx * in_features;
    for (int i = threadIdx.x; i < in_features; i += blockDim.x) {
        s_x[i] = x_sample[i];
    }
    __syncthreads();

    // Compute partial sum of (linear output - subtract) for assigned output features
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < out_features; j += blockDim.x) {
        float dot = has_bias ? bias[j] : 0.0f;
        const float* weight_row = weight + j * in_features;
        for (int i = 0; i < in_features; ++i) {
            dot += s_x[i] * weight_row[i];
        }
        dot -= subtract[j];
        local_sum += dot;
    }

    // Block reduction to compute total sum
    s_sum[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float total_sum = s_sum[0];
    float mean = total_sum / out_features;

    // GELU activation
    const float sqrt_2_over_pi = sqrtf(2.0f / 3.141592653589793f);
    float c = 0.044715f;
    float x_cube = mean * mean * mean;
    float inner = sqrt_2_over_pi * (mean + c * x_cube);
    float gelu_mean = 0.5f * mean * (1.0f + tanhf(inner));

    // Add to original x and write output
    float* out_sample = out + sample_idx * in_features;
    for (int i = threadIdx.x; i < in_features; i += blockDim.x) {
        out_sample[i] = s_x[i] + gelu_mean;
    }
}

torch::Tensor fused_linear_subtract_mean_gelu_residual_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor subtract)
{
    int batch_size = x.size(0);
    int in_features = x.size(1);
    int out_features = weight.size(0);

    auto out = torch::empty_like(x);

    const int threads = 256;
    const int blocks = batch_size;
    bool has_bias = bias.defined();

    int shared_mem_size = in_features * sizeof(float) + threads * sizeof(float);

    fused_kernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        has_bias ? bias.data_ptr<float>() : nullptr,
        subtract.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        has_bias
    );

    return out;
}
"""

fused_cpp_source = """
torch::Tensor fused_linear_subtract_mean_gelu_residual_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor subtract);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_linear_subtract_mean_gelu_residual",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_cuda_source,
    functions=["fused_linear_subtract_mean_gelu_residual_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        self.subtract = nn.Parameter(torch.randn(out_features))
        self.fused_op = fused_op

    def forward(self, x):
        # The custom kernel performs: linear, subtract, mean, GELU, and residual add in one pass
        return self.fused_op.fused_linear_subtract_mean_gelu_residual_cuda(
            x, self.weight, self.bias, self.subtract
        )