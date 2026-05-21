import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_M 16
#define TILE_N 16
#define TILE_K 32

__global__ void linear_sub_mul_relu_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int M, int K, int N,
    float subtract_value,
    float multiply_value
) {
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_M + ty;
    int col = blockIdx.x * TILE_N + tx;
    int tid = ty * TILE_N + tx;

    float acc = 0.0f;

    for (int base = 0; base < K; base += TILE_K) {
        for (int i = tid; i < TILE_M * TILE_K; i += TILE_M * TILE_N) {
            int r = i / TILE_K;
            int k = i - r * TILE_K;
            int gr = blockIdx.y * TILE_M + r;
            int gk = base + k;
            As[r][k] = (gr < M && gk < K) ? x[gr * K + gk] : 0.0f;
        }

        for (int i = tid; i < TILE_K * TILE_N; i += TILE_M * TILE_N) {
            int k = i / TILE_N;
            int c = i - k * TILE_N;
            int gc = blockIdx.x * TILE_N + c;
            int gk = base + k;
            Bs[k][c] = (gc < N && gk < K) ? weight[gc * K + gk] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        float v = (acc + bias[col] - subtract_value) * multiply_value;
        out[row * N + col] = v > 0.0f ? v : 0.0f;
    }
}

torch::Tensor linear_sub_mul_relu_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double subtract_value,
    double multiply_value
) {
    int M = (int)x.size(0);
    int K = (int)x.size(1);
    int N = (int)weight.size(0);

    auto out = torch::empty({M, N}, x.options());

    dim3 block(TILE_N, TILE_M);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    linear_sub_mul_relu_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        M, K, N,
        (float)subtract_value,
        (float)multiply_value
    );

    return out;
}
"""

cpp_sources = r"""
torch::Tensor linear_sub_mul_relu_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double subtract_value,
    double multiply_value
);
"""

linear_fused_ext = load_inline(
    name="linear_sub_mul_relu_ext",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["linear_sub_mul_relu_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        fan_in = in_features
        bound = 1 / (fan_in ** 0.5)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return linear_fused_ext.linear_sub_mul_relu_cuda(
            x, self.weight, self.bias, self.subtract_value, self.multiply_value
        )