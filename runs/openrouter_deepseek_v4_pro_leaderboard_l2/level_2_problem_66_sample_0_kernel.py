import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# CUDA source for fused linear + bias + dropout
linear_dropout_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define TILE_K 16

__global__ void linear_dropout_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    float dropout_p,
    bool training,
    unsigned long long seed
) {
    __shared__ float input_tile[16][16];
    __shared__ float weight_tile[16][16];

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float value = 0.0f;

    for (int k_tile = 0; k_tile < in_features; k_tile += TILE_K) {
        int input_row = blockIdx.y * blockDim.y + threadIdx.y;
        int input_col = k_tile + threadIdx.x;
        if (input_row < batch_size && input_col < in_features) {
            input_tile[threadIdx.y][threadIdx.x] = input[input_row * in_features + input_col];
        } else {
            input_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int weight_row = blockIdx.x * blockDim.x + threadIdx.x;
        int weight_col = k_tile + threadIdx.y;
        if (weight_row < out_features && weight_col < in_features) {
            weight_tile[threadIdx.x][threadIdx.y] = weight[weight_row * in_features + weight_col];
        } else {
            weight_tile[threadIdx.x][threadIdx.y] = 0.0f;
        }

        __syncthreads();

        if (row < batch_size && col < out_features) {
            for (int k = 0; k < TILE_K; ++k) {
                value += input_tile[threadIdx.y][k] * weight_tile[threadIdx.x][k];
            }
        }

        __syncthreads();
    }

    if (row < batch_size && col < out_features) {
        value += bias[col];

        if (training) {
            curandStatePhilox4_32_10_t state;
            curand_init(seed, 0, row * out_features + col, &state);
            float rand = curand_uniform(&state);
            if (rand < dropout_p) {
                value = 0.0f;
            } else {
                value = value / (1.0f - dropout_p);
            }
        }

        output[row * out_features + col] = value;
    }
}

torch::Tensor linear_dropout_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float dropout_p,
    bool training,
    unsigned long long seed
) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, input.options());

    const dim3 block_size(16, 16);
    const dim3 grid_size(
        (out_features + block_size.x - 1) / block_size.x,
        (batch_size + block_size.y - 1) / block_size.y
    );

    linear_dropout_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        dropout_p,
        training,
        seed
    );

    return output;
}
"""

linear_dropout_cpp_source = "torch::Tensor linear_dropout_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float dropout_p, bool training, unsigned long long seed);"

# Compile the inline CUDA code
linear_dropout = load_inline(
    name="linear_dropout",
    cpp_sources=linear_dropout_cpp_source,
    cuda_sources=linear_dropout_source,
    functions=["linear_dropout_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_p = dropout_p
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()
        self.linear_dropout = linear_dropout

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        seed = torch.randint(0, 2**31-1, (1,), device=x.device).item()
        training = self.training
        out = self.linear_dropout.linear_dropout_cuda(x, self.weight, self.bias, self.dropout_p, training, seed)
        out = torch.softmax(out, dim=1)
        return out