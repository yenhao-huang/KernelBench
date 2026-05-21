import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_avgpool_linear_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    int N,
    int C,
    int H,
    int W,
    int K
) {
    int nk = blockIdx.x;
    int n = nk / K;
    int k = nk - n * K;
    int tid = threadIdx.x;
    int HW = H * W;
    int total = C * HW;

    float acc = 0.0f;
    for (int i = tid; i < total; i += blockDim.x) {
        int c = i / HW;
        int hw = i - c * HW;
        acc += x[((n * C + c) * HW) + hw] * w[k * C + c];
    }

    __shared__ float smem[256];
    smem[tid] = acc;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float val = smem[0] / (float)HW;
        if (b != nullptr) val += b[k];
        out[n * K + k] = val;
    }
}

torch::Tensor fused_avgpool_linear_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int K = w.size(0);

    auto out = torch::empty({N, K}, x.options());
    int blocks = N * K;
    fused_avgpool_linear_kernel<<<blocks, 256>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        b.defined() ? b.data_ptr<float>() : nullptr,
        out.data_ptr<float>(),
        N, C, H, W, K
    );
    return out;
}
"""

cpp_sources = r"""
torch::Tensor fused_avgpool_linear_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b);
"""

_fused_ops = load_inline(
    name="resnet18_fused_avgpool_linear_ops",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["fused_avgpool_linear_cuda"],
    verbose=False,
)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        self.fused_ops = _fused_ops

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
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

        x = self.fused_ops.fused_avgpool_linear_cuda(x.contiguous(), self.fc.weight.contiguous(), self.fc.bias.contiguous())
        return x