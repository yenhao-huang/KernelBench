import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cpp_sources = """
torch::Tensor fused_model_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor subtract, bool has_bias);
"""

cuda_sources = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void reduce_params_kernel(
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ subtract,
    float* __restrict__ coeff,
    float* __restrict__ cst,
    int in_features,
    int out_features,
    bool has_bias
) {
    int col = blockIdx.x;
    __shared__ float smem[256];
    float sum = 0.0f;

    if (col < in_features) {
        for (int j = threadIdx.x; j < out_features; j += blockDim.x) {
            sum += weight[j * in_features + col];
        }
    } else {
        for (int j = threadIdx.x; j < out_features; j += blockDim.x) {
            float v = -subtract[j];
            if (has_bias) v += bias[j];
            sum += v;
        }
    }

    smem[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float v = smem[0] / (float)out_features;
        if (col < in_features) {
            coeff[col] = v;
        } else {
            cst[0] = v;
        }
    }
}

__device__ __forceinline__ float gelu_exact(float x) {
    const float inv_sqrt2 = 0.70710678118654752440f;
    return 0.5f * x * (1.0f + erff(x * inv_sqrt2));
}

__global__ void dot_gelu_residual_kernel(
    const float* __restrict__ x,
    const float* __restrict__ coeff,
    const float* __restrict__ cst,
    float* __restrict__ out,
    int batch,
    int in_features
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    __shared__ float smem[256];

    float sum = 0.0f;
    const float* row_x = x + row * in_features;

    for (int i = tid; i < in_features; i += blockDim.x) {
        sum += row_x[i] * coeff[i];
    }

    smem[tid] = sum;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    float addv = gelu_exact(smem[0] + cst[0]);

    for (int i = tid; i < in_features; i += blockDim.x) {
        out[row * in_features + i] = row_x[i] + addv;
    }
}

torch::Tensor fused_model_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor subtract, bool has_bias) {
    const int batch = (int)x.size(0);
    const int in_features = (int)x.size(1);
    const int out_features = (int)weight.size(0);

    auto coeff = torch::empty({in_features}, x.options());
    auto cst = torch::empty({1}, x.options());
    auto out = torch::empty_like(x);

    const int threads = 256;
    reduce_params_kernel<<<in_features + 1, threads>>>(
        weight.data_ptr<float>(),
        has_bias ? bias.data_ptr<float>() : nullptr,
        subtract.data_ptr<float>(),
        coeff.data_ptr<float>(),
        cst.data_ptr<float>(),
        in_features,
        out_features,
        has_bias
    );

    dot_gelu_residual_kernel<<<batch, threads>>>(
        x.data_ptr<float>(),
        coeff.data_ptr<float>(),
        cst.data_ptr<float>(),
        out.data_ptr<float>(),
        batch,
        in_features
    );

    return out;
}
"""

fused_model_ext = load_inline(
    name="kernelbench_fused_gemm_mean_lse_gelu_residual",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["fused_model_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        bias = self.gemm.bias if self.gemm.bias is not None else self.subtract
        return fused_model_ext.fused_model_cuda(
            x,
            self.gemm.weight,
            bias,
            self.subtract,
            self.gemm.bias is not None,
        )