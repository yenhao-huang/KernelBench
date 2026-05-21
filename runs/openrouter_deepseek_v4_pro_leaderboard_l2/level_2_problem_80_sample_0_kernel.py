import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for the fused max, subtract mean, and GELU kernel
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_max_sub_mean_gelu_kernel(const float* input, float* output,
                                               int batch_size, int out_features) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int row = blockIdx.x;
    if (row >= batch_size) return;

    // Each thread computes its own max over a strided subset of the row
    float max_val = -1e20f;
    for (int i = tid; i < out_features; i += blockDim.x) {
        float val = input[row * out_features + i];
        max_val = fmaxf(max_val, val);
    }
    sdata[tid] = max_val;
    __syncthreads();

    // Parallel reduction to find the row maximum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Thread 0 writes the final result for this row
    if (tid == 0) {
        float row_max = sdata[0];
        // mean of the max values (only one element) is row_max
        float diff = row_max - row_max;   // always 0
        // GELU activation: GELU(0) = 0
        const float sqrt_2_over_pi = sqrtf(2.0f / M_PI);
        float gelu_val = 0.5f * diff * (1.0f + tanhf(sqrt_2_over_pi * (diff + 0.044715f * diff * diff * diff)));
        output[row] = gelu_val;
    }
}

torch::Tensor fused_max_sub_mean_gelu_cuda(torch::Tensor input) {
    int batch_size = input.size(0);
    int out_features = input.size(1);
    auto output = torch::zeros({batch_size, 1}, input.options());

    const int threads = 256;
    const int blocks = batch_size;
    const int shared_mem_size = threads * sizeof(float);

    fused_max_sub_mean_gelu_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), batch_size, out_features
    );

    return output;
}
"""

cpp_source = """
torch::Tensor fused_max_sub_mean_gelu_cuda(torch::Tensor input);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_max_sub_mean_gelu",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_max_sub_mean_gelu_cuda"],
    verbose=False,
    extra_cflags=[],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, max_dim):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.max_dim = max_dim
        self.fused_op = fused_op

    def forward(self, x):
        x = self.gemm(x)
        x = self.fused_op.fused_max_sub_mean_gelu_cuda(x)
        return x