import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused activation operations
fused_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_activation_kernel(float* x, const float* add_value, int batch_size, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_features;
    if (idx < total) {
        int col = idx % out_features;
        float val = x[idx] + add_value[col];
        // Swish: sigmoid(val) * val
        val = (1.0f / (1.0f + expf(-val))) * val;
        // tanh
        val = tanhf(val);
        // GELU: 0.5 * val * (1.0 + erf(val / sqrt(2)))
        val = 0.5f * val * (1.0f + erff(val * 0.70710678118f));
        // Hardtanh: clamp to [-1, 1]
        val = fminf(fmaxf(val, -1.0f), 1.0f);
        x[idx] = val;
    }
}

torch::Tensor fused_activation_cuda(torch::Tensor x, torch::Tensor add_value) {
    int batch_size = x.size(0);
    int out_features = x.size(1);
    int total = batch_size * out_features;

    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;

    fused_activation_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), add_value.data_ptr<float>(), batch_size, out_features
    );

    return x;
}
"""

fused_activation_cpp_source = "torch::Tensor fused_activation_cuda(torch::Tensor x, torch::Tensor add_value);"

# Compile the inline CUDA code for fused activation
fused_activation = load_inline(
    name="fused_activation",
    cpp_sources=fused_activation_cpp_source,
    cuda_sources=fused_activation_source,
    functions=["fused_activation_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, add_value_shape):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape))
        self.fused_activation = fused_activation

    def forward(self, x):
        x = self.matmul(x)
        x = self.fused_activation.fused_activation_cuda(x, self.add_value)
        return x