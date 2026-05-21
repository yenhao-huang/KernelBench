import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define TILE 16

__global__ void linear_clamp_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int B, int K, int H,
    float scale2,
    float clamp_min,
    float clamp_max
) {
    __shared__ float xs[TILE][TILE];
    __shared__ float ws[TILE][TILE];

    int b = blockIdx.y * TILE + threadIdx.y;
    int h = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    for (int t = 0; t < K; t += TILE) {
        int kx = t + threadIdx.x;
        int ky = t + threadIdx.y;

        xs[threadIdx.y][threadIdx.x] = (b < B && kx < K) ? x[b * K + kx] : 0.0f;
        ws[threadIdx.y][threadIdx.x] = (h < H && ky < K) ? w[h * K + ky] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; ++i) {
            acc += xs[threadIdx.y][i] * ws[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (b < B && h < H) {
        float v = (acc + bias[h]) * scale2;
        v = fminf(fmaxf(v, clamp_min), clamp_max);
        y[b * H + h] = v;
    }
}

__global__ void lse_mish_kernel(
    const float* __restrict__ y,
    float* __restrict__ out,
    int B,
    int H
) {
    int b = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float smem[];
    float* smax = smem;
    float* ssum = smem + blockDim.x;

    float m = -INFINITY;
    for (int h = tid; h < H; h += blockDim.x) {
        m = fmaxf(m, y[b * H + h]);
    }

    smax[tid] = m;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smax[tid] = fmaxf(smax[tid], smax[tid + stride]);
        }
        __syncthreads();
    }

    float row_max = smax[0];
    float sum = 0.0f;

    for (int h = tid; h < H; h += blockDim.x) {
        sum += expf(y[b * H + h] - row_max);
    }

    ssum[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            ssum[tid] += ssum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float lse = logf(ssum[0]) + row_max;
        float sp = log1pf(expf(lse));
        float mish = lse * tanhf(sp);
        out[b] = lse * mish;
    }
}

torch::Tensor fused_linear_lse_mish_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor bias,
    double scale_factor,
    double clamp_min,
    double clamp_max
) {
    int B = x.size(0);
    int K = x.size(1);
    int H = w.size(0);

    auto temp = torch::empty({B, H}, x.options());
    auto out = torch::empty({B, 1}, x.options());

    dim3 block(TILE, TILE);
    dim3 grid((H + TILE - 1) / TILE, (B + TILE - 1) / TILE);

    float scale2 = static_cast<float>(scale_factor * 2.0);

    linear_clamp_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        bias.data_ptr<float>(),
        temp.data_ptr<float>(),
        B, K, H,
        scale2,
        static_cast<float>(clamp_min),
        static_cast<float>(clamp_max)
    );

    int reduce_threads = 256;
    lse_mish_kernel<<<B, reduce_threads, reduce_threads * 2 * sizeof(float)>>>(
        temp.data_ptr<float>(),
        out.data_ptr<float>(),
        B,
        H
    );

    return out;
}
"""

cpp_sources = r"""
torch::Tensor fused_linear_lse_mish_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor bias,
    double scale_factor,
    double clamp_min,
    double clamp_max
);
"""

fused_ops = load_inline(
    name="fused_linear_lse_mish_ops",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["fused_linear_lse_mish_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = float(scale_factor)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

    def forward(self, x):
        return fused_ops.fused_linear_lse_mish_cuda(
            x.contiguous(),
            self.matmul.weight.contiguous(),
            self.matmul.bias.contiguous(),
            self.scale_factor,
            self.clamp_min,
            self.clamp_max,
        )