import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source code for the fused kernel
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void gemm_multiply_leakyrelu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int M, int N, int K,
    float multiplier,
    float negative_slope
) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += input[row * K + k] * weight[col * K + k];
        }
        sum += bias[col];
        sum *= multiplier;
        output[row * N + col] = (sum > 0.0f) ? sum : (negative_slope * sum);
    }
}

torch::Tensor gemm_multiply_leakyrelu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float multiplier,
    float negative_slope
) {
    const int M = input.size(0);
    const int K = input.size(1);
    const int N = weight.size(0);
    
    auto output = torch::empty({M, N}, input.options());
    
    const int block_size = 256;
    const int grid_x = (N + block_size - 1) / block_size;
    const int grid_y = M;
    
    dim3 grid(grid_x, grid_y);
    dim3 block(block_size);
    
    gemm_multiply_leakyrelu_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K,
        multiplier,
        negative_slope
    );
    
    return output;
}
"""

cpp_source = "torch::Tensor gemm_multiply_leakyrelu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float multiplier, float negative_slope);"

# Compile the inline CUDA code
fused_op = load_inline(
    name="gemm_multiply_leakyrelu",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["gemm_multiply_leakyrelu_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.multiplier = multiplier
        self.negative_slope = negative_slope
        self.fused_op = fused_op

    def forward(self, x):
        # Use the fused CUDA kernel: GEMM + multiply + LeakyReLU
        return self.fused_op.gemm_multiply_leakyrelu_cuda(
            x,
            self.gemm.weight,
            self.gemm.bias,
            self.multiplier,
            self.negative_slope
        )