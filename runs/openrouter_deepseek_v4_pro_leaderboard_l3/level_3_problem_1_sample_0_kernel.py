import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

# CUDA source for fused linear + ReLU
linear_relu_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BM 64
#define BN 64
#define BK 8

__global__ void linear_relu_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K) {

    int bx = blockIdx.x;
    int by = blockIdx.y;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_base = by * BM + ty;
    int col_base = bx * BN + tx;

    int num_rows_per_thread = (BM + blockDim.y - 1) / blockDim.y;
    int num_cols_per_thread = (BN + blockDim.x - 1) / blockDim.x;

    float sums[4][4];
    for (int r = 0; r < num_rows_per_thread; ++r) {
        for (int c = 0; c < num_cols_per_thread; ++c) {
            sums[r][c] = 0.0f;
        }
    }

    for (int k_block = 0; k_block < K; k_block += BK) {
        // Load A tile into As
        for (int i = 0; i < BM * BK; i += blockDim.x * blockDim.y) {
            int idx = i + ty * blockDim.x + tx;
            if (idx < BM * BK) {
                int load_row = idx / BK;
                int load_col = idx % BK;
                int row = by * BM + load_row;
                int col = k_block + load_col;
                if (row < M && col < K) {
                    As[load_row][load_col] = A[row * K + col];
                } else {
                    As[load_row][load_col] = 0.0f;
                }
            }
        }

        // Load B tile into Bs
        for (int i = 0; i < BK * BN; i += blockDim.x * blockDim.y) {
            int idx = i + ty * blockDim.x + tx;
            if (idx < BK * BN) {
                int load_row = idx / BN;  // i
                int load_col = idx % BN;  // j
                int n = bx * BN + load_col;
                int k = k_block + load_row;
                if (n < N && k < K) {
                    Bs[load_row][load_col] = B[n * K + k];
                } else {
                    Bs[load_row][load_col] = 0.0f;
                }
            }
        }

        __syncthreads();

        for (int r = 0; r < num_rows_per_thread; ++r) {
            int row = row_base + r * blockDim.y;
            if (row >= M || row >= (by * BM + BM)) continue;
            for (int c = 0; c < num_cols_per_thread; ++c) {
                int col = col_base + c * blockDim.x;
                if (col >= N || col >= (bx * BN + BN)) continue;
                float sum = sums[r][c];
                for (int k = 0; k < BK; ++k) {
                    sum += As[row - by * BM][k] * Bs[k][col - bx * BN];
                }
                sums[r][c] = sum;
            }
        }

        __syncthreads();
    }

    for (int r = 0; r < num_rows_per_thread; ++r) {
        int row = row_base + r * blockDim.y;
        if (row >= M) continue;
        for (int c = 0; c < num_cols_per_thread; ++c) {
            int col = col_base + c * blockDim.x;
            if (col >= N) continue;
            float val = sums[r][c] + bias[col];
            val = val > 0 ? val : 0;
            C[row * N + col] = val;
        }
    }
}

torch::Tensor linear_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    int M = input.size(0);
    int K = input.size(1);
    int N = weight.size(0);
    auto output = torch::empty({M, N}, input.options());

    const dim3 blockDim(16, 16);
    const dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    linear_relu_kernel<<<gridDim, blockDim>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K
    );

    return output;
}
"""

linear_relu_cpp_source = "torch::Tensor linear_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"

# Compile the inline CUDA code
linear_relu = load_inline(
    name="linear_relu",
    cpp_sources=linear_relu_cpp_source,
    cuda_sources=linear_relu_cuda_source,
    functions=["linear_relu_cuda"],
    verbose=False,
    extra_cflags=[],
    extra_ldflags=[],
)

class LinearReLU(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return linear_relu.linear_relu_cuda(x, self.weight, self.bias)

class ModelNew(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        super(ModelNew, self).__init__()

        layers = []
        current_input_size = input_size

        for layer_size in layer_sizes:
            layers.append(LinearReLU(current_input_size, layer_size))
            current_input_size = layer_size

        layers.append(nn.Linear(current_input_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)