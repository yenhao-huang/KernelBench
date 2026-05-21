import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

avgpool_linear_cpp_source = """
torch::Tensor avgpool_linear_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias);
"""

avgpool_linear_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avgpool_linear_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int N,
    int C,
    int H,
    int W,
    int K
) {
    int k = blockIdx.x;
    int n = blockIdx.y;
    int tid = threadIdx.x;
    int spatial = H * W;
    int total = C * spatial;

    float acc = 0.0f;
    const float inv_spatial = 1.0f / (float)spatial;
    const int x_base = n * total;

    for (int idx = tid; idx < total; idx += blockDim.x) {
        int c = idx / spatial;
        acc += x[x_base + idx] * weight[k * C + c] * inv_spatial;
    }

    __shared__ float smem[256];
    smem[tid] = acc;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float v = smem[0];
        if (bias != nullptr) {
            v += bias[k];
        }
        out[n * K + k] = v;
    }
}

torch::Tensor avgpool_linear_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    int N = (int)x.size(0);
    int C = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);
    int K = (int)weight.size(0);

    auto out = torch::empty({N, K}, x.options());

    dim3 grid(K, N);
    dim3 block(256);

    avgpool_linear_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        out.data_ptr<float>(),
        N, C, H, W, K
    );

    return out;
}
"""

avgpool_linear_ext = load_inline(
    name="efficientnet_b1_avgpool_linear_ext",
    cpp_sources=avgpool_linear_cpp_source,
    cuda_sources=avgpool_linear_cuda_source,
    functions=["avgpool_linear_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.mbconv1 = self._make_mbconv_block(32, 16, 1, 1)
        self.mbconv2 = self._make_mbconv_block(16, 24, 2, 6)
        self.mbconv3 = self._make_mbconv_block(24, 40, 2, 6)
        self.mbconv4 = self._make_mbconv_block(40, 80, 2, 6)
        self.mbconv5 = self._make_mbconv_block(80, 112, 1, 6)
        self.mbconv6 = self._make_mbconv_block(112, 192, 2, 6)
        self.mbconv7 = self._make_mbconv_block(192, 320, 1, 6)

        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)

        self.fc = nn.Linear(1280, num_classes)
        self.avgpool_linear = avgpool_linear_ext

    def _make_mbconv_block(self, in_channels, out_channels, stride, expand_ratio):
        hidden_dim = round(in_channels * expand_ratio)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))

        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)
        x = self.mbconv6(x)
        x = self.mbconv7(x)

        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.avgpool_linear.avgpool_linear_cuda(x, self.fc.weight, self.fc.bias)
        return x