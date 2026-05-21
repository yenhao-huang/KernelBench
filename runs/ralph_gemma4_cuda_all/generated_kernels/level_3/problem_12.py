import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused ReLU and Dropout (identity during inference)
# This kernel performs element-wise ReLU. During inference, Dropout(p=0.0) is identity.
# Fusing them reduces kernel launch overhead and memory bandwidth usage.
fused_relu_dropout_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_relu_dropout_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = data[idx];
        data[idx] = val > 0.0f ? val : 0.0f;
    }
}

void fused_relu_dropout_cuda(torch::Tensor x) {
    auto size = x.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_relu_dropout_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), size);
}
"""

fused_relu_dropout_cpp_source = """
void fused_relu_dropout_cuda(torch::Tensor x);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_relu_dropout_cpp_source,
    cuda_sources=fused_relu_dropout_source,
    functions=["fused_relu_dropout_cuda"],
    verbose=False,
)

class FusedReLUDropout(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training or self.p > 0:
            # Fallback to standard PyTorch if training or p > 0
            return F.dropout(F.relu(x), p=self.p, training=self.training)
        else:
            # During inference with p=0, use fused kernel
            # We use inplace to save memory, matching the original's inplace=True
            fused_ops.fused_relu_dropout_cuda(x)
            return x

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        
        # We replace the standard ReLU(inplace=True) and Dropout(p=0.0) 
        # with our FusedReLUDropout module.
        # Note: Conv2d and MaxPool2d are highly optimized in cuDNN, 
        # so we keep them as is.
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            FusedReLUDropout(p=0.0),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            FusedReLUDropout(p=0.0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            FusedReLUDropout(p=0.0),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            FusedReLUDropout(p=0.0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            FusedReLUDropout(p=0.0),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            FusedReLUDropout(p=0.0),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            FusedReLUDropout(p=0.0),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            FusedReLUDropout(p=0.0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            FusedReLUDropout(p=0.0),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            FusedReLUDropout(p=0.0),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            FusedReLUDropout(p=0.0),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            FusedReLUDropout(p=0.0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            FusedReLUDropout(p=0.0),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            FusedReLUDropout(p=0.0),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            FusedReLUDropout(p=0.0),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            FusedReLUDropout(p=0.0),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            FusedReLUDropout(p=0.0),
            nn.Dropout(p=0.0),
            nn.Linear(4096, 4096),
            FusedReLUDropout(p=0.0),
            nn.Dropout(p=0.0),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x