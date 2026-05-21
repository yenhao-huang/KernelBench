import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cpp_sources = """
torch::Tensor linear_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias);
torch::Tensor groupnorm_leaky_double_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, int num_groups, double eps, double negative_slope);
"""

cuda_sources = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define TILE 16

__global__ void linear_kernel(const float* __restrict__ x,
                              const float* __restrict__ w,
                              const float* __restrict__ b,
                              float* __restrict__ out,
                              int M, int K, int N) {
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
        out[row * N + col] = acc + b[col];
    }
}

__global__ void groupnorm_leaky_double_kernel(const float* __restrict__ x,
                                              const float* __restrict__ gamma,
                                              const float* __restrict__ beta,
                                              float* __restrict__ out,
                                              int M, int C, int G,
                                              float eps, float negative_slope) {
    int row = blockIdx.x;
    int group = blockIdx.y;
    int tid = threadIdx.x;
    int group_size = C / G;
    int base = row * C + group * group_size;

    float sum = 0.0f;
    for (int i = tid; i < group_size; i += blockDim.x) {
        sum += x[base + i];
    }

    __shared__ float sh[32];
    sh[tid] = sum;
    __syncthreads();

    for (int stride = 16; stride > 0; stride >>= 1) {
        if (tid < stride) sh[tid] += sh[tid + stride];
        __syncthreads();
    }

    float mean = sh[0] / (float)group_size;

    float sq = 0.0f;
    for (int i = tid; i < group_size; i += blockDim.x) {
        float d = x[base + i] - mean;
        sq += d * d;
    }

    sh[tid] = sq;
    __syncthreads();

    for (int stride = 16; stride > 0; stride >>= 1) {
        if (tid < stride) sh[tid] += sh[tid + stride];
        __syncthreads();
    }

    float inv_std = rsqrtf(sh[0] / (float)group_size + eps);

    for (int i = tid; i < group_size; i += blockDim.x) {
        int c = group * group_size + i;
        float y = (x[base + i] - mean) * inv_std;
        y = y * gamma[c] + beta[c];
        y = y > 0.0f ? y : y * negative_slope;
        out[base + i] = y + y;
    }
}

torch::Tensor linear_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    int M = x.size(0);
    int K = x.size(1);
    int N = weight.size(0);

    auto out = torch::empty({M, N}, x.options());

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    linear_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        M, K, N
    );

    return out;
}

torch::Tensor groupnorm_leaky_double_cuda(torch::Tensor x,
                                          torch::Tensor gamma,
                                          torch::Tensor beta,
                                          int num_groups,
                                          double eps,
                                          double negative_slope) {
    int M = x.size(0);
    int C = x.size(1);

    auto out = torch::empty_like(x);

    dim3 block(32);
    dim3 grid(M, num_groups);

    groupnorm_leaky_double_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        out.data_ptr<float>(),
        M, C, num_groups,
        (float)eps,
        (float)negative_slope
    );

    return out;
}
"""

kernelbench_ops = load_inline(
    name="kernelbench_linear_gn_leaky_double_ops",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["linear_cuda", "groupnorm_leaky_double_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super(ModelNew, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size, eps=eps)
        self.num_groups = num_groups
        self.eps = eps
        self.negative_slope = negative_slope
        self.ops = kernelbench_ops

    def forward(self, x):
        x = self.ops.linear_cuda(x, self.fc.weight, self.fc.bias)
        x = self.ops.groupnorm_leaky_double_cuda(
            x, self.gn.weight, self.gn.bias, self.num_groups, self.eps, self.negative_slope
        )
        return x