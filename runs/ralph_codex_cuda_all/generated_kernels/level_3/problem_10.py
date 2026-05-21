import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void bn_relu_kernel(const float* __restrict__ x, const float* __restrict__ w,
                               const float* __restrict__ b, const float* __restrict__ mean,
                               const float* __restrict__ var, float* __restrict__ y,
                               int total, int C, int HW, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        int c = (idx / HW) % C;
        float v = (x[idx] - mean[c]) * rsqrtf(var[c] + eps) * w[c] + b[c];
        y[idx] = v > 0.0f ? v : 0.0f;
    }
}

__global__ void bn_kernel(const float* __restrict__ x, const float* __restrict__ w,
                          const float* __restrict__ b, const float* __restrict__ mean,
                          const float* __restrict__ var, float* __restrict__ y,
                          int total, int C, int HW, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        int c = (idx / HW) % C;
        y[idx] = (x[idx] - mean[c]) * rsqrtf(var[c] + eps) * w[c] + b[c];
    }
}

__global__ void add_relu_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                float* __restrict__ y, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        float v = a[idx] + b[idx];
        y[idx] = v > 0.0f ? v : 0.0f;
    }
}

__global__ void maxpool3x3s2p1_kernel(const float* __restrict__ x, float* __restrict__ y,
                                      int N, int C, int H, int W, int OH, int OW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * OH * OW;
    if (idx < total) {
        int ow = idx % OW;
        int oh = (idx / OW) % OH;
        int c = (idx / (OW * OH)) % C;
        int n = idx / (OW * OH * C);
        float m = -FLT_MAX;
        int base_h = oh * 2 - 1;
        int base_w = ow * 2 - 1;
        for (int kh = 0; kh < 3; ++kh) {
            int ih = base_h + kh;
            if (ih >= 0 && ih < H) {
                for (int kw = 0; kw < 3; ++kw) {
                    int iw = base_w + kw;
                    if (iw >= 0 && iw < W) {
                        float v = x[((n * C + c) * H + ih) * W + iw];
                        m = v > m ? v : m;
                    }
                }
            }
        }
        y[idx] = m;
    }
}

__global__ void avgpool_kernel(const float* __restrict__ x, float* __restrict__ y,
                               int N, int C, int HW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C;
    if (idx < total) {
        const float* p = x + idx * HW;
        float s = 0.0f;
        for (int i = 0; i < HW; ++i) s += p[i];
        y[idx] = s / (float)HW;
    }
}

__global__ void linear_kernel(const float* __restrict__ x, const float* __restrict__ w,
                              const float* __restrict__ b, float* __restrict__ y,
                              int N, int K, int O) {
    int o = blockIdx.x;
    int n = blockIdx.y;
    int tid = threadIdx.x;
    __shared__ float smem[256];
    float acc = 0.0f;
    for (int k = tid; k < K; k += blockDim.x) {
        acc += x[n * K + k] * w[o * K + k];
    }
    smem[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) y[n * O + o] = smem[0] + b[o];
}

torch::Tensor bn_relu_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b,
                           torch::Tensor mean, torch::Tensor var, double eps) {
    auto y = torch::empty_like(x);
    int total = x.numel();
    int C = x.size(1);
    int HW = x.size(2) * x.size(3);
    int threads = 256;
    bn_relu_kernel<<<(total + threads - 1) / threads, threads>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(),
        mean.data_ptr<float>(), var.data_ptr<float>(), y.data_ptr<float>(),
        total, C, HW, (float)eps);
    return y;
}

torch::Tensor bn_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b,
                      torch::Tensor mean, torch::Tensor var, double eps) {
    auto y = torch::empty_like(x);
    int total = x.numel();
    int C = x.size(1);
    int HW = x.size(2) * x.size(3);
    int threads = 256;
    bn_kernel<<<(total + threads - 1) / threads, threads>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(),
        mean.data_ptr<float>(), var.data_ptr<float>(), y.data_ptr<float>(),
        total, C, HW, (float)eps);
    return y;
}

torch::Tensor add_relu_cuda(torch::Tensor a, torch::Tensor b) {
    auto y = torch::empty_like(a);
    int total = a.numel();
    int threads = 256;
    add_relu_kernel<<<(total + threads - 1) / threads, threads>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), total);
    return y;
}

torch::Tensor maxpool_cuda(torch::Tensor x) {
    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    int OH = (H + 1) / 2;
    int OW = (W + 1) / 2;
    auto y = torch::empty({N, C, OH, OW}, x.options());
    int total = N * C * OH * OW;
    int threads = 256;
    maxpool3x3s2p1_kernel<<<(total + threads - 1) / threads, threads>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), N, C, H, W, OH, OW);
    return y;
}

torch::Tensor avgpool_cuda(torch::Tensor x) {
    int N = x.size(0), C = x.size(1), HW = x.size(2) * x.size(3);
    auto y = torch::empty({N, C}, x.options());
    int total = N * C;
    int threads = 256;
    avgpool_kernel<<<(total + threads - 1) / threads, threads>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), N, C, HW);
    return y;
}

torch::Tensor linear_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
    int N = x.size(0), K = x.size(1), O = w.size(0);
    auto y = torch::empty({N, O}, x.options());
    linear_kernel<<<dim3(O, N), 256>>>(x.data_ptr<float>(), w.data_ptr<float>(),
                                       b.data_ptr<float>(), y.data_ptr<float>(), N, K, O);
    return y;
}
"""

cpp_sources = r"""
torch::Tensor bn_relu_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor mean, torch::Tensor var, double eps);
torch::Tensor bn_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor mean, torch::Tensor var, double eps);
torch::Tensor add_relu_cuda(torch::Tensor a, torch::Tensor b);
torch::Tensor maxpool_cuda(torch::Tensor x);
torch::Tensor avgpool_cuda(torch::Tensor x);
torch::Tensor linear_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b);
"""

_ops = load_inline(
    name="resnet50_fused_ops",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["bn_relu_cuda", "bn_cuda", "add_relu_cuda", "maxpool_cuda", "avgpool_cuda", "linear_cuda"],
    verbose=False,
)


class BottleneckNew(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def _bn_relu(self, x, bn):
        return _ops.bn_relu_cuda(x, bn.weight, bn.bias, bn.running_mean, bn.running_var, bn.eps)

    def _bn(self, x, bn):
        return _ops.bn_cuda(x, bn.weight, bn.bias, bn.running_mean, bn.running_var, bn.eps)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self._bn_relu(out, self.bn1)

        out = self.conv2(out)
        out = self._bn_relu(out, self.bn2)

        out = self.conv3(out)
        out = self._bn(out, self.bn3)

        if self.downsample is not None:
            identity = self.downsample[0](x)
            identity = self._bn(identity, self.downsample[1])

        return _ops.add_relu_cuda(out, identity)


class ModelNew(nn.Module):
    def __init__(self, layers, num_classes=1000):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        block = BottleneckNew
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = _ops.bn_relu_cuda(x, self.bn1.weight, self.bn1.bias, self.bn1.running_mean, self.bn1.running_var, self.bn1.eps)
        x = _ops.maxpool_cuda(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = _ops.avgpool_cuda(x)
        x = _ops.linear_cuda(x, self.fc.weight, self.fc.bias)
        return x