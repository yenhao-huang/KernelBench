import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cpp_sources = """
torch::Tensor linear_relu_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b);
torch::Tensor linear_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b);
"""

cuda_sources = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE 16

__global__ void linear_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    int M, int K, int N,
    int do_relu
) {
    __shared__ float xs[TILE][TILE];
    __shared__ float ws[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    for (int t = 0; t < K; t += TILE) {
        int x_col = t + threadIdx.x;
        int w_col = t + threadIdx.y;

        xs[threadIdx.y][threadIdx.x] = (row < M && x_col < K) ? x[row * K + x_col] : 0.0f;
        ws[threadIdx.y][threadIdx.x] = (col < N && w_col < K) ? w[col * K + w_col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; ++i) {
            acc += xs[threadIdx.y][i] * ws[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        acc += b[col];
        if (do_relu && acc < 0.0f) acc = 0.0f;
        out[row * N + col] = acc;
    }
}

torch::Tensor linear_relu_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
    int M = x.size(0);
    int K = x.size(1);
    int N = w.size(0);

    auto out = torch::empty({M, N}, x.options());

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    linear_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        M, K, N, 1
    );

    return out;
}

torch::Tensor linear_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
    int M = x.size(0);
    int K = x.size(1);
    int N = w.size(0);

    auto out = torch::empty({M, N}, x.options());

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    linear_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        M, K, N, 0
    );

    return out;
}
"""

mlp_cuda = load_inline(
    name="kernelbench_mlp_linear_relu_fp32",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["linear_relu_cuda", "linear_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        super().__init__()

        sizes = [input_size] + list(layer_sizes) + [output_size]
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        for in_features, out_features in zip(sizes[:-1], sizes[1:]):
            weight = nn.Parameter(torch.empty(out_features, in_features))
            bias = nn.Parameter(torch.empty(out_features))
            nn.init.kaiming_uniform_(weight, a=5 ** 0.5)
            fan_in = in_features
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(bias, -bound, bound)
            self.weights.append(weight)
            self.biases.append(bias)

        self.ops = mlp_cuda

    def forward(self, x):
        for i in range(len(self.weights) - 1):
            x = self.ops.linear_relu_cuda(x, self.weights[i], self.biases[i])
        return self.ops.linear_cuda(x, self.weights[-1], self.biases[-1])