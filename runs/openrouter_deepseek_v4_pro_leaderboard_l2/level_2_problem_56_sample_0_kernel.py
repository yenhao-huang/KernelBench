import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused linear + sigmoid + sum
fused_linear_sigmoid_sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_N 32
#define TILE_K 32

__global__ void fused_linear_sigmoid_sum_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int input_size,
    int hidden_size) {

    int b = blockIdx.x;
    int j_block = blockIdx.y;
    int j_start = j_block * TILE_N;
    int j_end = min(j_start + TILE_N, hidden_size);

    int tx = threadIdx.x; // j index within tile
    int ty = threadIdx.y; // k index within tile

    __shared__ float x_shared[TILE_K];
    __shared__ float W_shared[TILE_N][TILE_K];

    float acc = 0.0f;

    // Iterate over k tiles
    for (int k_block = 0; k_block < (input_size + TILE_K - 1) / TILE_K; ++k_block) {
        int k_start = k_block * TILE_K;
        int k_end = min(k_start + TILE_K, input_size);

        // Load x tile into shared memory
        if (ty < k_end - k_start) {
            x_shared[ty] = x[b * input_size + k_start + ty];
        } else {
            x_shared[ty] = 0.0f;
        }

        // Load W tile into shared memory
        if (tx < j_end - j_start && ty < k_end - k_start) {
            W_shared[tx][ty] = weight[(j_start + tx) * input_size + k_start + ty];
        } else if (tx < j_end - j_start) {
            W_shared[tx][ty] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        if (tx < j_end - j_start) {
            for (int k = 0; k < k_end - k_start; ++k) {
                acc += x_shared[k] * W_shared[tx][k];
            }
        }

        __syncthreads();
    }

    // Apply bias and sigmoid
    float sig_val = 0.0f;
    if (tx < j_end - j_start) {
        float val = acc + bias[j_start + tx];
        sig_val = 1.0f / (1.0f + expf(-val));
    }

    // Block reduction to sum sig_val across tx
    __shared__ float sum_shared[TILE_N];
    sum_shared[tx] = sig_val;
    __syncthreads();

    for (int stride = TILE_N / 2; stride > 0; stride >>= 1) {
        if (tx < stride) {
            sum_shared[tx] += sum_shared[tx + stride];
        }
        __syncthreads();
    }

    // Thread 0 adds the block's partial sum to global output
    if (tx == 0 && ty == 0) {
        atomicAdd(&output[b], sum_shared[0]);
    }
}

torch::Tensor fused_linear_sigmoid_sum_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias) {

    int batch_size = x.size(0);
    int input_size = x.size(1);
    int hidden_size = weight.size(0);

    auto output = torch::zeros({batch_size, 1}, x.options());

    dim3 block(TILE_N, TILE_K);
    dim3 grid(batch_size, (hidden_size + TILE_N - 1) / TILE_N);

    fused_linear_sigmoid_sum_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input_size,
        hidden_size
    );

    return output;
}
"""

fused_linear_sigmoid_sum_cpp_source = "torch::Tensor fused_linear_sigmoid_sum_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias);"

# Compile the inline CUDA code
fused_linear_sigmoid_sum = load_inline(
    name="fused_linear_sigmoid_sum",
    cpp_sources=fused_linear_sigmoid_sum_cpp_source,
    cuda_sources=fused_linear_sigmoid_sum_source,
    functions=["fused_linear_sigmoid_sum_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.fused_op = fused_linear_sigmoid_sum

    def forward(self, x):
        return self.fused_op.fused_linear_sigmoid_sum_cuda(
            x.contiguous(),
            self.linear.weight.contiguous(),
            self.linear.bias.contiguous()
        )