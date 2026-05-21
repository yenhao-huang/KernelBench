import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Combined CUDA source for matmul+GELU and softmax
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// GELU activation function
__device__ float gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f; // sqrt(2/pi)
    const float coeff = 0.044715f;
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// Tiled matrix multiplication + GELU kernel
// Computes out = GELU(x @ weight.T + bias)
// x: [M, K], weight: [N, K] (stored as [N, K] for linear layer), bias: [N]
// out: [M, N]
__global__ void matmul_gelu_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int M, int N, int K) {

    const int TILE_M = 32;
    const int TILE_N = 32;
    const int TILE_K = 8;

    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;

    // Shared memory for tiles
    __shared__ float x_tile[TILE_M][TILE_K];
    __shared__ float w_tile[TILE_K][TILE_N];

    for (int k = 0; k < K; k += TILE_K) {
        // Load x tile: threads with threadIdx.x < TILE_K load
        if (threadIdx.x < TILE_K) {
            int x_col = k + threadIdx.x;
            if (x_col < K) {
                x_tile[threadIdx.y][threadIdx.x] = x[row * K + x_col];
            } else {
                x_tile[threadIdx.y][threadIdx.x] = 0.0f;
            }
        }
        // Load weight tile: threads with threadIdx.y < TILE_K load
        if (threadIdx.y < TILE_K) {
            int w_row = k + threadIdx.y;
            if (w_row < K) {
                // weight is [N, K], so weight[col][w_row]
                w_tile[threadIdx.y][threadIdx.x] = weight[col * K + w_row];
            } else {
                w_tile[threadIdx.y][threadIdx.x] = 0.0f;
            }
        }
        __syncthreads();

        // Compute partial dot product
        for (int i = 0; i < TILE_K; i++) {
            sum += x_tile[threadIdx.y][i] * w_tile[i][threadIdx.x];
        }
        __syncthreads();
    }

    // Add bias and apply GELU
    float val = sum + bias[col];
    out[row * N + col] = gelu(val);
}

// Wrapper for matmul+GELU
torch::Tensor matmul_gelu_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    int M = x.size(0);
    int K = x.size(1);
    int N = weight.size(0); // weight is [N, K]

    auto out = torch::zeros({M, N}, x.options());

    const int TILE_M = 32;
    const int TILE_N = 32;
    dim3 block(TILE_N, TILE_M);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    matmul_gelu_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        M, N, K);

    return out;
}

// Softmax kernel (row-wise)
// Each block processes one row
__global__ void softmax_kernel(const float* __restrict__ input, float* __restrict__ output, int N) {
    extern __shared__ float shared[];
    // shared memory layout:
    // [0, blockDim.x) for max reduction
    // [blockDim.x, 2*blockDim.x) for sum reduction

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Step 1: find max
    float local_max = -INFINITY;
    for (int i = tid; i < N; i += stride) {
        float val = input[row * N + i];
        if (val > local_max) local_max = val;
    }
    shared[tid] = local_max;
    __syncthreads();

    // Reduction for max
    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared[tid + s] > shared[tid]) {
                shared[tid] = shared[tid + s];
            }
        }
        __syncthreads();
    }
    float row_max = shared[0];
    __syncthreads();

    // Step 2: compute exp sum
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += stride) {
        float val = expf(input[row * N + i] - row_max);
        local_sum += val;
    }
    shared[tid] = local_sum;
    __syncthreads();

    // Reduction for sum
    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    float row_sum = shared[0];
    __syncthreads();

    // Step 3: normalize and write output
    for (int i = tid; i < N; i += stride) {
        float val = expf(input[row * N + i] - row_max) / row_sum;
        output[row * N + i] = val;
    }
}

// Wrapper for softmax
torch::Tensor softmax_cuda(torch::Tensor input) {
    int M = input.size(0);
    int N = input.size(1);
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    int shared_mem_size = 2 * block_size * sizeof(float);
    softmax_kernel<<<M, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N);

    return output;
}
"""

cpp_source = """
torch::Tensor matmul_gelu_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias);
torch::Tensor softmax_cuda(torch::Tensor input);
"""

# Compile the inline CUDA code
custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["matmul_gelu_cuda", "softmax_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.matmul_gelu = custom_ops.matmul_gelu_cuda
        self.softmax = custom_ops.softmax_cuda

    def forward(self, x):
        # Custom fused matmul + GELU
        x = self.matmul_gelu(x, self.linear.weight, self.linear.bias)
        # Custom softmax
        x = self.softmax(x)
        return x