import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

#define TILE 16

__global__ void linear_sigmoid_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    int M, int K, int N
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    for (int t = 0; t < K; t += TILE) {
        int ak = t + threadIdx.x;
        int bk = t + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && ak < K) ? x[row * K + ak] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (col < N && bk < K) ? w[col * K + bk] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; ++i) {
            acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        acc += b[col];
        out[row * N + col] = 1.0f / (1.0f + expf(-acc));
    }
}

__global__ void linear_logsumexp_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    int M, int K, int N
) {
    extern __shared__ float vals[];

    int row = blockIdx.x;
    int col = threadIdx.x;

    float v = -FLT_MAX;

    if (row < M && col < N) {
        float acc = 0.0f;
        const float* xr = x + row * K;
        const float* wr = w + col * K;

        for (int k = 0; k < K; ++k) {
            acc += xr[k] * wr[k];
        }

        v = acc + b[col];
    }

    vals[col] = v;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (col < stride) {
            vals[col] = fmaxf(vals[col], vals[col + stride]);
        }
        __syncthreads();
    }

    float maxv = vals[0];

    float e = (col < N) ? expf(v - maxv) : 0.0f;
    vals[col] = e;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (col < stride) {
            vals[col] += vals[col + stride];
        }
        __syncthreads();
    }

    if (col == 0 && row < M) {
        out[row] = logf(vals[0]) + maxv;
    }
}

torch::Tensor linear_sigmoid_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
    int M = x.size(0);
    int K = x.size(1);
    int N = w.size(0);

    auto out = torch::empty({M, N}, x.options());

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    linear_sigmoid_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        M, K, N
    );

    return out;
}

torch::Tensor linear_logsumexp_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
    int M = x.size(0);
    int K = x.size(1);
    int N = w.size(0);

    auto out = torch::empty({M}, x.options());

    linear_logsumexp_kernel<<<M, 1024, 1024 * sizeof(float)>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        M, K, N
    );

    return out;
}
"""

cpp_sources = r"""
torch::Tensor linear_sigmoid_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b);
torch::Tensor linear_logsumexp_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b);
"""

ops = load_inline(
    name="kb_gemm_sigmoid_gemm_lse_ops",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["linear_sigmoid_cuda", "linear_logsumexp_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.ops = ops

    def forward(self, x):
        h = self.ops.linear_sigmoid_cuda(
            x.contiguous(),
            self.linear1.weight.contiguous(),
            self.linear1.bias.contiguous(),
        )
        return self.ops.linear_logsumexp_cuda(
            h,
            self.linear2.weight.contiguous(),
            self.linear2.bias.contiguous(),
        )