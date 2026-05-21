import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool_kernel(const float* __restrict__ x, float* __restrict__ pooled,
                                int N, int C, int H, int W) {
    int idx = blockIdx.x;
    int n = idx / C;
    int c = idx - n * C;
    int HW = H * W;

    float sum = 0.0f;
    const float* base = x + ((n * C + c) * HW);

    for (int i = threadIdx.x; i < HW; i += blockDim.x) {
        sum += base[i];
    }

    __shared__ float smem[256];
    smem[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        pooled[n * C + c] = smem[0] / (float)HW;
    }
}

__global__ void linear_kernel(const float* __restrict__ pooled,
                              const float* __restrict__ weight,
                              const float* __restrict__ bias,
                              float* __restrict__ out,
                              int N, int C, int O) {
    int idx = blockIdx.x;
    int n = idx / O;
    int o = idx - n * O;

    float sum = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        sum += pooled[n * C + c] * weight[o * C + c];
    }

    __shared__ float smem[256];
    smem[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out[n * O + o] = smem[0] + bias[o];
    }
}

torch::Tensor avgpool_linear_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    int N = (int)x.size(0);
    int C = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);
    int O = (int)weight.size(0);

    auto pooled = torch::empty({N, C}, x.options());
    auto out = torch::empty({N, O}, x.options());

    const int threads = 256;
    avg_pool_kernel<<<N * C, threads>>>(x.data_ptr<float>(), pooled.data_ptr<float>(), N, C, H, W);
    linear_kernel<<<N * O, threads>>>(pooled.data_ptr<float>(), weight.data_ptr<float>(),
                                      bias.data_ptr<float>(), out.data_ptr<float>(), N, C, O);
    return out;
}
"""

cpp_sources = "torch::Tensor avgpool_linear_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias);"

effnet_ops = load_inline(
    name="effnet_b0_avgpool_linear_ops",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["avgpool_linear_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(MBConv, self).__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        hidden_dim = in_channels * expand_ratio

        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            )

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        )

        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        identity = x
        if hasattr(self, "expand_conv"):
            x = self.expand_conv(x)
        x = self.depthwise_conv(x)
        x = self.project_conv(x)
        if self.use_residual:
            x = x + identity
        return x


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.blocks = nn.Sequential(
            MBConv(32, 16, kernel_size=3, stride=1, expand_ratio=1),
            MBConv(16, 24, kernel_size=3, stride=2, expand_ratio=6),
            MBConv(24, 24, kernel_size=3, stride=1, expand_ratio=6),
            MBConv(24, 40, kernel_size=5, stride=2, expand_ratio=6),
            MBConv(40, 40, kernel_size=5, stride=1, expand_ratio=6),
            MBConv(40, 80, kernel_size=3, stride=2, expand_ratio=6),
            MBConv(80, 80, kernel_size=3, stride=1, expand_ratio=6),
            MBConv(80, 112, kernel_size=5, stride=1, expand_ratio=6),
            MBConv(112, 112, kernel_size=5, stride=1, expand_ratio=6),
            MBConv(112, 192, kernel_size=5, stride=2, expand_ratio=6),
            MBConv(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            MBConv(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            MBConv(192, 320, kernel_size=3, stride=1, expand_ratio=6),
        )

        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(1280, num_classes)
        self.ops = effnet_ops

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.ops.avgpool_linear_cuda(x.contiguous(), self.fc.weight.contiguous(), self.fc.bias.contiguous())
        return x