import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused (bias + y) * y
fused_add_mul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_add_mul_kernel(const float* y, const float* bias, float* out, int batch_size, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_features;
    if (idx < total) {
        int feat = idx % out_features;
        float y_val = y[idx];
        out[idx] = (bias[feat] + y_val) * y_val;
    }
}

torch::Tensor fused_add_mul_cuda(torch::Tensor y, torch::Tensor bias) {
    auto batch_size = y.size(0);
    auto out_features = y.size(1);
    auto out = torch::empty_like(y);
    int total = batch_size * out_features;
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;
    fused_add_mul_kernel<<<num_blocks, block_size>>>(
        y.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), batch_size, out_features
    );
    return out;
}
"""

fused_add_mul_cpp_source = "torch::Tensor fused_add_mul_cuda(torch::Tensor y, torch::Tensor bias);"

# Compile the inline CUDA code
fused_add_mul = load_inline(
    name="fused_add_mul",
    cpp_sources=fused_add_mul_cpp_source,
    cuda_sources=fused_add_mul_source,
    functions=["fused_add_mul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that replaces the original forward with a fused CUDA kernel.
    The linear and instance norm are kept for parameter compatibility but not used.
    """
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.bmm = nn.Linear(in_features, out_features)
        self.instance_norm = nn.InstanceNorm2d(out_features, eps=eps, momentum=momentum)
        self.fused_add_mul = fused_add_mul

    def forward(self, x, y):
        # The original computation reduces to (instance_norm.bias + y) * y
        # because InstanceNorm2d on a 1x1 spatial input outputs only the bias.
        return self.fused_add_mul.fused_add_mul_cuda(y, self.instance_norm.bias)

# Input generation functions (unchanged)
batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    return [torch.rand(batch_size, in_features), torch.rand(batch_size, out_features)]

def get_init_inputs():
    return [in_features, out_features]