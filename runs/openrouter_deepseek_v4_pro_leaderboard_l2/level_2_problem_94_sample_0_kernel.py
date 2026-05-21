import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused bias addition, Hardtanh, and Mish
fused_bias_hardtanh_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void bias_hardtanh_mish_kernel(const float* input, const float* bias, float* output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        int col = idx % cols;
        float val = input[idx] + bias[col];
        // Hardtanh: clamp to [-1, 1]
        val = fminf(fmaxf(val, -1.0f), 1.0f);
        // Mish: val * tanh(softplus(val))
        float sp;
        if (val > 20.0f) {
            sp = val;
        } else {
            sp = log1pf(expf(val));
        }
        output[idx] = val * tanhf(sp);
    }
}

torch::Tensor bias_hardtanh_mish_cuda(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must be 2D (batch, features)");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
    TORCH_CHECK(input.size(1) == bias.size(0), "input feature dim must match bias size");

    int rows = input.size(0);
    int cols = input.size(1);
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (rows * cols + block_size - 1) / block_size;

    bias_hardtanh_mish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), rows, cols);

    return output;
}
"""

fused_bias_hardtanh_mish_cpp_source = "torch::Tensor bias_hardtanh_mish_cuda(torch::Tensor input, torch::Tensor bias);"

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_bias_hardtanh_mish",
    cpp_sources=fused_bias_hardtanh_mish_cpp_source,
    cuda_sources=fused_bias_hardtanh_mish_source,
    functions=["bias_hardtanh_mish_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model with fused bias+Hardtanh+Mish CUDA kernel.
    """
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)  # bias handled in custom kernel
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)
        self.fused_op = fused_op

    def forward(self, x):
        x = self.gemm(x)                              # GEMM
        x = self.fused_op.bias_hardtanh_mish_cuda(x, self.bias)  # Fused bias+Hardtanh+Mish
        x = self.groupnorm(x)                         # GroupNorm
        return x


# Helper functions (same interface as original)
batch_size = 1024
in_features = 8192
out_features = 8192
bias_shape = (out_features,)
num_groups = 256

def get_inputs():
    return [torch.randn(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, bias_shape, num_groups]