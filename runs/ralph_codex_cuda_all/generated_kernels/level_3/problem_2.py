import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE 16

__global__ void linear_t_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w_t,
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
        int w_row = t + threadIdx.y;

        xs[threadIdx.y][threadIdx.x] = (row < M && x_col < K) ? x[row * K + x_col] : 0.0f;
        ws[threadIdx.y][threadIdx.x] = (w_row < K && col < N) ? w_t[w_row * N + col] : 0.0f;

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

torch::Tensor linear_t_cuda(torch::Tensor x, torch::Tensor w_t, torch::Tensor b, bool do_relu) {
    int M = (int)x.size(0);
    int K = (int)x.size(1);
    int N = (int)w_t.size(1);

    auto out = torch::empty({M, N}, x.options());

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    linear_t_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        w_t.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        M, K, N,
        do_relu ? 1 : 0
    );

    return out;
}
"""

cpp_sources = r"""
torch::Tensor linear_t_cuda(torch::Tensor x, torch::Tensor w_t, torch::Tensor b, bool do_relu);
"""

linear_t_ext = load_inline(
    name="kb_mlp_linear_t_ext",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["linear_t_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super().__init__()

        sizes = [input_size] + list(hidden_layer_sizes) + [output_size]
        self.weights_t = nn.ParameterList()
        self.biases = nn.ParameterList()

        for in_features, out_features in zip(sizes[:-1], sizes[1:]):
            w_t = nn.Parameter(torch.empty(in_features, out_features))
            b = nn.Parameter(torch.empty(out_features))
            bound = 1.0 / math.sqrt(in_features)
            nn.init.uniform_(w_t, -bound, bound)
            nn.init.uniform_(b, -bound, bound)
            self.weights_t.append(w_t)
            self.biases.append(b)

        self.op = linear_t_ext

    def forward(self, x):
        for i in range(len(self.weights_t)):
            x = self.op.linear_t_cuda(
                x.contiguous(),
                self.weights_t[i].contiguous(),
                self.biases[i].contiguous(),
                i + 1 < len(self.weights_t),
            )
        return x