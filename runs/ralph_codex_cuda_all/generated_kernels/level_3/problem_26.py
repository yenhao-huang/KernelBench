import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void channel_shuffle_kernel(const float* __restrict__ x, float* __restrict__ y,
                                       int total, int C, int H, int W, int groups) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int spatial = H * W;
    int s = idx % spatial;
    int c_out = (idx / spatial) % C;
    int n = idx / (spatial * C);

    int cpg = C / groups;
    int group = c_out % groups;
    int inner = c_out / groups;
    int c_in = group * cpg + inner;

    y[idx] = x[((n * C + c_in) * spatial) + s];
}

torch::Tensor channel_shuffle_cuda(torch::Tensor x, int64_t groups) {
    auto y = torch::empty_like(x);
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int total = N * C * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    channel_shuffle_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(),
                                                total, C, H, W, (int)groups);
    return y;
}

__global__ void avgpool_linear_kernel(const float* __restrict__ x,
                                      const float* __restrict__ weight,
                                      const float* __restrict__ bias,
                                      float* __restrict__ y,
                                      int N, int C, int H, int W, int K) {
    int k = blockIdx.x;
    int n = blockIdx.y;
    int tid = threadIdx.x;
    int spatial = H * W;
    int count = C * spatial;

    __shared__ float partial[256];
    float sum = 0.0f;

    for (int i = tid; i < count; i += blockDim.x) {
        int c = i / spatial;
        float v = x[(n * C * spatial) + i];
        sum += v * weight[k * C + c];
    }

    partial[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }

    if (tid == 0) {
        y[n * K + k] = partial[0] / (float)spatial + bias[k];
    }
}

torch::Tensor avgpool_linear_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int K = weight.size(0);
    auto y = torch::empty({N, K}, x.options());
    dim3 blocks(K, N);
    avgpool_linear_kernel<<<blocks, 256>>>(x.data_ptr<float>(), weight.data_ptr<float>(),
                                           bias.data_ptr<float>(), y.data_ptr<float>(),
                                           N, C, H, W, K);
    return y;
}
"""

cpp_sources = r"""
torch::Tensor channel_shuffle_cuda(torch::Tensor x, int64_t groups);
torch::Tensor avgpool_linear_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias);
"""

shufflenet_ops = load_inline(
    name="shufflenet_custom_ops",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["channel_shuffle_cuda", "avgpool_linear_cuda"],
    verbose=False,
)


class ChannelShuffleNew(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        return shufflenet_ops.channel_shuffle_cuda(x.contiguous(), self.groups)


class ShuffleNetUnitNew(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        super().__init__()
        assert out_channels % 4 == 0
        mid_channels = out_channels // 4

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.shuffle = ChannelShuffleNew(groups)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.shuffle(out)
        out = self.relu(self.bn3(self.conv3(out)))
        out = out + self.shortcut(x)
        return out


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000, groups=3, stages_repeats=[3, 7, 3], stages_out_channels=[24, 240, 480, 960]):
        super().__init__()

        self.conv1 = nn.Conv2d(3, stages_out_channels[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(stages_out_channels[0])
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage2 = self._make_stage(stages_out_channels[0], stages_out_channels[1], stages_repeats[0], groups)
        self.stage3 = self._make_stage(stages_out_channels[1], stages_out_channels[2], stages_repeats[1], groups)
        self.stage4 = self._make_stage(stages_out_channels[2], stages_out_channels[3], stages_repeats[2], groups)

        self.conv5 = nn.Conv2d(stages_out_channels[3], 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(1024)
        self.fc = nn.Linear(1024, num_classes)

    def _make_stage(self, in_channels, out_channels, repeats, groups):
        layers = [ShuffleNetUnitNew(in_channels, out_channels, groups)]
        for _ in range(1, repeats):
            layers.append(ShuffleNetUnitNew(out_channels, out_channels, groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.relu(self.bn5(self.conv5(x)))
        x = shufflenet_ops.avgpool_linear_cuda(x.contiguous(), self.fc.weight, self.fc.bias)
        return x