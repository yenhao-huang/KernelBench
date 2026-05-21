import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline


# Define the custom CUDA kernel for fused add + ReLU
add_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void add_relu_kernel(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = a[idx] + b[idx];
        out[idx] = val > 0.0f ? val : 0.0f;
    }
}

torch::Tensor add_relu_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have the same shape");
    auto out = torch::empty_like(a);
    int n = a.numel();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    add_relu_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}
"""

add_relu_cpp_source = "torch::Tensor add_relu_cuda(torch::Tensor a, torch::Tensor b);"

# Compile the inline CUDA code for add_relu
add_relu = load_inline(
    name="add_relu",
    cpp_sources=add_relu_cpp_source,
    cuda_sources=add_relu_source,
    functions=["add_relu_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )
        self.stride = stride
        self.add_relu = add_relu  # custom fused kernel

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # Fused element-wise addition + ReLU
        out = self.add_relu.add_relu_cuda(out, identity)
        return out