import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source code for the fused kernel
fused_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_kernel(const float* input, float* output, int batch_size, int out_features) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const float* row_input = input + row * out_features;

    // Step 1: find max value in the row
    float max_val = -INFINITY;
    for (int i = tid; i < out_features; i += blockDim.x) {
        float val = row_input[i];
        if (val > max_val) max_val = val;
    }
    shared[tid] = max_val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared[tid + s] > shared[tid]) {
                shared[tid] = shared[tid + s];
            }
        }
        __syncthreads();
    }
    float row_max = shared[0];
    __syncthreads();

    // Step 2: compute sum of exp(x - max)
    float sum_exp = 0.0f;
    for (int i = tid; i < out_features; i += blockDim.x) {
        sum_exp += expf(row_input[i] - row_max);
    }
    shared[tid] = sum_exp;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    float total_sum = shared[0];
    __syncthreads();

    // Step 3: apply activations (only thread 0)
    if (tid == 0) {
        float logsumexp = logf(total_sum) + row_max;

        // LeakyReLU (negative_slope = 0.01) twice
        float lr1 = logsumexp > 0.0f ? logsumexp : 0.01f * logsumexp;
        float lr2 = lr1 > 0.0f ? lr1 : 0.01f * lr1;

        // GELU (exact) twice
        const float sqrt2 = 1.41421356237f;
        float gelu1 = lr2 * 0.5f * (1.0f + erff(lr2 / sqrt2));
        float gelu2 = gelu1 * 0.5f * (1.0f + erff(gelu1 / sqrt2));

        output[row] = gelu2;
    }
}

torch::Tensor fused_logsumexp_leakyrelu_gelu_cuda(torch::Tensor input) {
    int batch_size = input.size(0);
    int out_features = input.size(1);
    auto output = torch::empty({batch_size, 1}, input.options());

    const int block_size = 256;
    const int grid_size = batch_size;
    size_t shared_mem_size = block_size * sizeof(float);

    fused_kernel<<<grid_size, block_size, shared_mem_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), batch_size, out_features
    );

    return output;
}
"""

fused_cpp_source = "torch::Tensor fused_logsumexp_leakyrelu_gelu_cuda(torch::Tensor input);"

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_cuda_source,
    functions=["fused_logsumexp_leakyrelu_gelu_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.fused_op = fused_ops

    def forward(self, x):
        x = self.linear(x)
        x = self.fused_op.fused_logsumexp_leakyrelu_gelu_cuda(x)
        return x