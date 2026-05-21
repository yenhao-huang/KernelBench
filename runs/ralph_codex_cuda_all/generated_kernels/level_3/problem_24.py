import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

linear4d_cpp_source = """
torch::Tensor linear4d_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias);
"""

linear4d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void linear4d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    int N,
    int C,
    int K
) {
    extern __shared__ float smem[];
    int nk = blockIdx.x;
    int n = nk / K;
    int k = nk - n * K;
    int tid = threadIdx.x;

    float sum = 0.0f;
    const float* xrow = x + n * C;
    const float* wrow = w + k * C;

    for (int c = tid; c < C; c += blockDim.x) {
        sum += xrow[c] * wrow[c];
    }

    smem[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[n * K + k] = smem[0] + b[k];
    }
}

torch::Tensor linear4d_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    int N = x.size(0);
    int C = x.size(1);
    int K = weight.size(0);

    auto out = torch::empty({N, K}, x.options());

    const int threads = 256;
    const int blocks = N * K;
    linear4d_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        C,
        K
    );

    return out;
}
"""

linear4d_ext = load_inline(
    name="efficientnet_b2_linear4d_ext",
    cpp_sources=linear4d_cpp_source,
    cuda_sources=linear4d_cuda_source,
    functions=["linear4d_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.mbconv1 = self._make_mbconv_block(32, 96, 1, 3)
        self.mbconv2 = self._make_mbconv_block(96, 144, 2, 6)
        self.mbconv3 = self._make_mbconv_block(144, 192, 2, 6)
        self.mbconv4 = self._make_mbconv_block(192, 288, 2, 6)
        self.mbconv5 = self._make_mbconv_block(288, 384, 1, 6)

        self.conv_final = nn.Conv2d(384, 1408, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_final = nn.BatchNorm2d(1408)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1408, num_classes)

        self.linear4d_ext = linear4d_ext

    def _make_mbconv_block(self, in_channels, out_channels, stride, expand_ratio):
        layers = []
        expanded_channels = in_channels * expand_ratio

        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, expanded_channels, kernel_size=1, stride=1, padding=0, bias=False))
            layers.append(nn.BatchNorm2d(expanded_channels))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3, stride=stride, padding=1, groups=expanded_channels, bias=False))
        layers.append(nn.BatchNorm2d(expanded_channels))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Conv2d(expanded_channels, expanded_channels // 4, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(expanded_channels // 4, expanded_channels, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.Sigmoid())

        layers.append(nn.Conv2d(expanded_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)
        x = self.relu(self.bn_final(self.conv_final(x)))
        x = self.linear4d_ext.linear4d_cuda(x.contiguous(), self.fc.weight.contiguous(), self.fc.bias.contiguous())
        return x