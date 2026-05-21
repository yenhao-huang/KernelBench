import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for BatchNorm2d inference
# This kernel performs: out = (x - mean) / sqrt(var + eps) * weight + bias
# It is optimized for memory bandwidth by fusing the subtraction, division, multiplication, and addition.
bn_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void batchnorm2d_inference_kernel(
    const float* __restrict__ input,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W,
    float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * H * W;

    if (idx < total_elements) {
        // Calculate channel index
        // idx = n*C*H*W + c*H*W + h*W + w
        int c = (idx / (H * W)) % C;
        
        float m = running_mean[c];
        float v = running_var[c];
        float w_val = weight[c];
        float b_val = bias[c];
        
        float inv_std = 1.0f / sqrtf(v + eps);
        
        output[idx] = (input[idx] - m) * inv_std * w_val + b_val;
    }
}

torch::Tensor batchnorm2d_inference_cuda(
    torch::Tensor input,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps) {
    
    auto output = torch::empty_like(input);
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int total_elements = N * C * H * W;

    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    batchnorm2d_inference_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        eps
    );

    return output;
}
"""

bn_cpp_source = """
torch::Tensor batchnorm2d_inference_cuda(
    torch::Tensor input,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps);
"""

# Compile the inline CUDA code
bn_cuda_module = load_inline(
    name="bn_cuda_module",
    cpp_sources=bn_cpp_source,
    cuda_sources=bn_cuda_source,
    functions=["batchnorm2d_inference_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Batch Normalization using a custom CUDA kernel.
    Note: This implementation focuses on the inference (eval) mode for maximum speedup
    via kernel fusion.
    """
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features)
        self.num_features = num_features
        self.bn_cuda = bn_cuda_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If in training mode, use standard PyTorch BatchNorm2d as it requires 
        # calculating batch statistics (mean/var) which is complex to fuse 
        # with the update logic in a single simple kernel.
        if self.training:
            return self.bn(x)
        
        # In eval mode, we use our fused CUDA kernel
        return self.bn_cuda.batchnorm2d_inference_cuda(
            x,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.weight,
            self.bn.bias,
            self.bn.eps
        )