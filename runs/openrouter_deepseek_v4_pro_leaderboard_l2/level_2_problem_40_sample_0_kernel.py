import torch
import torch.nn as nn
import torch.nn.init as init
import math
from torch.utils.cpp_extension import load_inline

# CUDA source for fused bias addition and scaling
fused_bias_scale_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_bias_scale_kernel(const float* input, const float* bias, float* output,
                                        int batch_size, int out_features, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_features;
    if (idx < total_elements) {
        int col = idx % out_features;
        output[idx] = (input[idx] + bias[col]) * scale;
    }
}

torch::Tensor fused_bias_scale_cuda(torch::Tensor input, torch::Tensor bias, float scale) {
    auto batch_size = input.size(0);
    auto out_features = input.size(1);
    auto output = torch::empty_like(input);

    const int block_size = 256;
    int total_elements = batch_size * out_features;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_bias_scale_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, out_features, scale);

    return output;
}
"""

fused_bias_scale_cpp_source = "torch::Tensor fused_bias_scale_cuda(torch::Tensor input, torch::Tensor bias, float scale);"

# Compile the inline CUDA code
fused_bias_scale = load_inline(
    name="fused_bias_scale",
    cpp_sources=fused_bias_scale_cpp_source,
    cuda_sources=fused_bias_scale_cuda_source,
    functions=["fused_bias_scale_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that fuses bias addition and scaling into a single CUDA kernel.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scaling_factor = scaling_factor

        # Weight and bias parameters (same shape as nn.Linear)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        # Custom CUDA operator
        self.fused_bias_scale = fused_bias_scale

        # Initialize parameters like nn.Linear
        self.reset_parameters()

    def reset_parameters(self):
        # Same initialization as nn.Linear
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Matrix multiplication (no bias yet)
        matmul_result = torch.matmul(x, self.weight.t())  # (batch_size, out_features)

        # Fused bias addition and scaling: out = (matmul_result + bias) * (1 + scaling_factor)
        scale = 1.0 + self.scaling_factor
        out = self.fused_bias_scale.fused_bias_scale_cuda(matmul_result, self.bias, scale)

        return out