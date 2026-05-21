import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define TILE 16

__device__ __forceinline__ float gelu_exact(float x) {
    const float inv_sqrt2 = 0.70710678118654752440f;
    return 0.5f * x * (1.0f + erff(x * inv_sqrt2));
}

__global__ void linear_bn_gelu_relu_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    const float* __restrict__ bn_weight,
    const float* __restrict__ bn_bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float* __restrict__ out,
    int M,
    int K,
    int N,
    float eps
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    for (int t = 0; t < K; t += TILE) {
        int a_col = t + threadIdx.x;
        int b_col = t + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? x[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (col < N && b_col < K) ? w[col * K + b_col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; ++i) {
            acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        float y = acc + bias[col];
        y = (y - running_mean[col]) * rsqrtf(running_var[col] + eps) * bn_weight[col] + bn_bias[col];
        y = gelu_exact(y);
        y = y > 0.0f ? y : 0.0f;
        out[row * N + col] = y;
    }
}

torch::Tensor linear_bn_gelu_relu_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor bias,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps
) {
    int M = (int)x.size(0);
    int K = (int)x.size(1);
    int N = (int)w.size(0);

    auto out = torch::empty({M, N}, x.options());

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    linear_bn_gelu_relu_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        bias.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        out.data_ptr<float>(),
        M, K, N, (float)eps
    );

    return out;
}
"""

cpp_sources = r"""
torch::Tensor linear_bn_gelu_relu_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor bias,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps
);
"""

linear_bn_gelu_relu_ext = load_inline(
    name="linear_bn_gelu_relu_ext",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["linear_bn_gelu_relu_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)
        self.op = linear_bn_gelu_relu_ext

    def forward(self, x):
        return self.op.linear_bn_gelu_relu_cuda(
            x,
            self.gemm.weight,
            self.gemm.bias,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.eps,
        )