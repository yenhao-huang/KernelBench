```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Combined CUDA source for fused linear+Swish+bias and GroupNorm
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Tiled matrix multiplication with Swish activation and extra bias addition
#define TILE_M 32
#define TILE_N 32
#define TILE_K 32

__global__ void fused_linear_swish_bias_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias_lin,
    const float* __restrict__ bias_extra,
    float* __restrict__ out,
    int N, int in_features, int out_features)
{
    __shared__ float x_tile[TILE_M][TILE_K];
    __shared__ float w_tile[TILE_N][TILE_K];

    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    float sum = 0.0f;

    for (int k_block = 0; k_block < in_features; k_block += TILE_K) {
        // Load x tile into shared memory
        if (row < N && (k_block + threadIdx.x) < in_features) {
            x_tile[threadIdx.y][threadIdx.x] = x[row * in_features + k_block + threadIdx.x];
        } else {
            x_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load weight tile into shared memory
        if (col < out_features && (k_block + threadIdx.y) < in_features) {
            w_tile[threadIdx.x][threadIdx.y] = weight[col * in_features + k_block + threadIdx.y];
        } else {
            w_tile[threadIdx.x][threadIdx.y] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            sum += x_tile[threadIdx.y][k] * w_tile[threadIdx.x][k];
        }

        __syncthreads();
    }

    if (row < N && col < out_features) {
        sum += bias_lin[col];
        float sig = 1.0f / (1.0f + expf(-sum));
        sum = sum * sig + bias_extra[col];
        out[row * out_features + col] = sum;
    }
}

torch::Tensor fused_linear_swish_bias_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias_lin,
    torch::Tensor bias_extra)
{
    int N = x.size(0);
    int in_features = x.size(1);
    int out_features = weight.size(0);

    auto out = torch::empty({N, out_features}, x.options());

    const dim3 block(TILE_N, TILE_M);
    const dim3 grid((out_features + TILE_N - 1) / TILE_N,
                    (N + TILE_M - 1) / TILE_M);

    fused_linear_swish_bias_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_lin.data_ptr<float>(),
        bias_extra.data_ptr<float>(),
        out.data_ptr<float>(),
        N, in_features, out_features);

    return out;
}

// GroupNorm kernel: one block per (sample, group)
__global__ void group_norm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int N, int C, int num_groups, float eps)
{
    int group = blockIdx.x;
    int n = blockIdx.y;
    int C_per_group = C / num_groups;
    int tid = threadIdx.x;
    int c = group * C_per_group + tid;

    float val = input[n * C + c];

    // Shared memory for reduction (size = C_per_group)
    extern __shared__ float s_data[];

    // Compute sum
    s_data[tid] = val;
    __syncthreads();

    for (int stride = C_per_group / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }
    float sum = s_data[0];
    float mean = sum / C_per_group;

    // Compute variance
    float diff = val - mean;
    s_data[tid] = diff * diff;
    __syncthreads();

    for (int stride = C_per_group / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }
    float var = s_data[0] / C_per_group;

    float inv_std = rsqrtf(var + eps);

    // Normalize and apply affine parameters
    float normalized = diff * inv_std;
    output[n * C + c] = normalized * gamma[c] + beta[c];
}

torch::Tensor group_norm_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    int num_groups,
    float eps)
{
    int N = input.size(0);
    int C = input.size(1);
    int C_per_group = C / num_groups;

    auto output = torch::empty_like(input);

    const dim3 block(C_per_group);
    const dim3 grid(num_groups, N);

    size_t shared_mem_size = C_per_group * sizeof(float);

    group_norm_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, num_groups, eps);

    return output;
}
"""

cpp_source = """
torch::Tensor fused_linear_swish_bias_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias_lin,
    torch::Tensor bias_extra);

torch::Tensor group_norm_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    int num_groups,
    float eps);
"""

# Compile the inline CUDA code
custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_linear_swish_bias_cuda", "group_norm_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.num_groups = num_groups
        self.custom_ops = custom_ops

    def forward(self, x):
        # F