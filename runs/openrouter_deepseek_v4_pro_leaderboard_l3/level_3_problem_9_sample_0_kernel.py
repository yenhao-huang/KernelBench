import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------------
# Fused add + ReLU inline CUDA kernel
# ------------------------------------------------------------------
add_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void add_relu_kernel(float* out, const float* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = out[idx] + b[idx];
        out[idx] = (val > 0.0f) ? val : 0.0f;
    }
}

torch::Tensor add_relu_cuda(torch::Tensor out, torch::Tensor b) {
    auto size = out.numel();
    TORCH_CHECK(b.numel() == size, "Tensor size mismatch in add_relu_cuda");
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    add_relu_kernel<<<num_blocks, block_size>>>(out.data_ptr<float>(), b.data_ptr<float>(), size);
    return out;
}
"""

add_relu_cpp_source = "torch::Tensor add_relu_cuda(torch::Tensor out, torch::Tensor b);"

# Compile the custom operator
add_relu_module = load_inline(
    name="add_relu",
    cpp_sources=add_relu_cpp_source,
    cuda_sources=add_relu_source,
    functions=["add_relu_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""]
)

# ------------------------------------------------------------------
# Modified BasicBlock using the fused add+ReLU
# ------------------------------------------------------------------
class BasicBlockNew(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlockNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.add_relu = add_relu_module

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # In-place addition + ReLU via custom CUDA kernel
        out = self.add_relu.add_relu_cuda(out, identity)

        return out


# ------------------------------------------------------------------
# ResNet-like model using the optimized block (ModelNew)
# ------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlockNew, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlockNew, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlockNew, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlockNew, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlockNew.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x