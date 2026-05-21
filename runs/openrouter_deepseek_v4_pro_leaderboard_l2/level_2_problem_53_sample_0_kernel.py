import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for fused linear + scale + hardtanh + GELU
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__global__ void fused_linear_scale_hardtanh_gelu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    float scaling_factor,
    float hardtanh_min,
    float hardtanh_max)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_features;
    if (idx < total_elements) {
        int row = idx / out_features;
        int col = idx % out_features;

        // Compute dot product for this output element
        float sum = bias[col];
        const float* input_row = input + row * in_features;
        const float* weight_row = weight + col * in_features;
        for (int k = 0; k < in_features; ++k) {
            sum += input_row[k] * weight_row[k];
        }

        // Scale
        sum *= scaling_factor;

        // Hardtanh
        sum = fminf(fmaxf(sum, hardtanh_min), hardtanh_max);

        // GELU activation (tanh approximation)
        float x = sum;
        float x3 = x * x * x;
        float inner = sqrtf(2.0f / M_PI) * (x + 0.044715f * x3);
        float gelu = 0.5f * x * (1.0f + tanhf(inner));
        output[idx] = gelu;
    }
}

torch::Tensor fused_linear_scale_hardtanh_gelu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaling_factor,
    float hardtanh_min,
    float hardtanh_max)
{
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must be 2D (batch, in_features)");

    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, input.options());

    int total_elements = batch_size * out_features;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_linear_scale_hardtanh_gelu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        scaling_factor,
        hardtanh_min,
        hardtanh_max
    );

    return output;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_linear_scale_hardtanh_gelu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaling_factor,
    float hardtanh_min,
    float hardtanh_max);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_linear_scale_hardtanh_gelu",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_linear_scale_hardtanh_gelu_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        # Keep the linear layer to hold weight and bias parameters
        self.gemm = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self.fused_op = fused_op

    def forward(self, x):
        # Use the fused CUDA operator instead of separate steps
        return self.fused_op.fused_linear_scale_hardtanh_gelu_cuda(
            x,
            self.gemm.weight,
            self.gemm.bias,
            self.scaling_factor,
            self.hardtanh_min,
            self.hardtanh_max
        )