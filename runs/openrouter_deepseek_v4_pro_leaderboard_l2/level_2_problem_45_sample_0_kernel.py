import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source code for both custom kernels
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel 1: Fused linear (matrix multiply) + sigmoid
__global__ void fused_linear_sigmoid_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int batch,
    int input_size,
    int hidden_size)
{
    const int BLOCK_N = 64;
    const int BK = 16;

    int row = blockIdx.y;
    int col_start = blockIdx.x * BLOCK_N;
    int col = col_start + threadIdx.x;

    __shared__ float x_tile[BK];
    __shared__ float w_tile[BLOCK_N * BK];

    float accum = 0.0f;

    for (int k_tile = 0; k_tile < input_size; k_tile += BK) {
        // Load x tile into shared memory
        if (threadIdx.x < BK) {
            int k = k_tile + threadIdx.x;
            x_tile[threadIdx.x] = (row < batch && k < input_size) ? x[row * input_size + k] : 0.0f;
        }

        // Load weight tile into shared memory
        if (col < hidden_size) {
            for (int k = 0; k < BK; ++k) {
                int k_idx = k_tile + k;
                w_tile[threadIdx.x * BK + k] = (k_idx < input_size) ? weight[col * input_size + k_idx] : 0.0f;
            }
        } else {
            for (int k = 0; k < BK; ++k) {
                w_tile[threadIdx.x * BK + k] = 0.0f;
            }
        }

        __syncthreads();

        // Compute partial dot product
        if (col < hidden_size) {
            for (int k = 0; k < BK; ++k) {
                accum += x_tile[k] * w_tile[threadIdx.x * BK + k];
            }
        }

        __syncthreads();
    }

    // Add bias and apply sigmoid
    if (col < hidden_size) {
        float val = accum + bias[col];
        y[row * hidden_size + col] = 1.0f / (1.0f + expf(-val));
    }
}

// Kernel 2: Fused linear (matrix multiply) + logsumexp over features
__global__ void fused_linear_logsumexp_kernel(
    const float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ z,
    int batch,
    int hidden_size,
    int output_size)
{
    const int BK = 16;

    int row = blockIdx.x;
    int j = threadIdx.x;  // output feature index

    __shared__ float y_tile[BK];
    __shared__ float w_tile[1024 * BK];  // output_size = 1024, fixed
    __shared__ float sdata[1024];

    float accum = 0.0f;

    for (int k_tile = 0; k_tile < hidden_size; k_tile += BK) {
        // Load y tile
        if (threadIdx.x < BK) {
            int k = k_tile + threadIdx.x;
            y_tile[threadIdx.x] = (row < batch && k < hidden_size) ? y[row * hidden_size + k] : 0.0f;
        }

        // Load weight tile
        for (int k = 0; k < BK; ++k) {
            int k_idx = k_tile + k;
            w_tile[j * BK + k] = (k_idx < hidden_size) ? weight[j * hidden_size + k_idx] : 0.0f;
        }

        __syncthreads();

        // Partial dot product
        for (int k = 0; k < BK; ++k) {
            accum += y_tile[k] * w_tile[j * BK + k];
        }

        __syncthreads();
    }

    // Add bias
    float val = accum + bias[j];
    float my_val = val;

    // Store for reduction
    sdata[j] = my_val;
    __syncthreads();

    // Max reduction
    for (int stride = output_size / 2; stride > 0; stride >>= 1) {
        if (j < stride) {
            sdata[j] = fmaxf(sdata[j], sdata[j + stride]);
        }
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    // Exp and sum reduction
    sdata[j] = expf(my_val - max_val);
    __syncthreads();

    for (int stride = output_size / 2; stride > 0; stride >>= 1) {
        if (j < stride) {
            sdata[j] += sdata[j + stride];
        }
        __syncthreads();
    }

    if (j == 0) {
        z[row] = max_val + logf(sdata[0]);
    }
}

// Wrapper functions
torch::Tensor fused_linear_sigmoid_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    auto batch = x.size(0);
    auto input_size = x.size(1);
    auto hidden_size = weight.size(0);
    auto y = torch::empty({batch, hidden_size}, x.options());

    const int BLOCK_N = 64;
    dim3 block(BLOCK_N);
    dim3 grid((hidden_size + BLOCK_N - 1) / BLOCK_N, batch);

    fused_linear_sigmoid_kernel<<<grid, block>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), y.data_ptr<float>(),
        batch, input_size, hidden_size
    );
    return y;
}

torch::Tensor fused_linear_logsumexp_cuda(torch::Tensor y, torch::Tensor weight, torch::Tensor bias) {
    auto batch = y.size(0);
    auto hidden_size = y.size(1);
    auto output_size = weight.size(0);
    auto z = torch::empty({batch}, y.options());

    dim3 block(output_size);
    dim3 grid(batch);

    fused_linear_logsumexp_kernel<<<grid, block>>>(
        y.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), z.data_ptr<float>(),
        batch, hidden_size, output_size
    );
    return z;
}
"""

cpp_source = """
torch::Tensor fused_linear_sigmoid_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias);
torch::Tensor fused_linear_logsumexp_cuda(torch::Tensor y, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_linear_sigmoid_cuda", "fused_linear_logsumexp_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        # Keep original linear layers for weight storage
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.fused_ops = fused_ops

    def forward(self, x):
        # Fused linear1 + sigmoid
        x = self.fused_ops.fused_linear_sigmoid_cuda(x, self.linear1.weight, self.linear1.bias)
        # Fused linear2 + logsumexp
        x = self.fused_ops.fused_linear_logsumexp_cuda(x, self.linear2.weight, self.linear2.bias)
        return x