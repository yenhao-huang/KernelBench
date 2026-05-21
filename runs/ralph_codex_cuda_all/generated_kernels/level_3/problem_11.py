import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void relu_inplace_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        x[i] = v > 0.0f ? v : 0.0f;
    }
}

__global__ void maxpool2x2s2_kernel(const float* __restrict__ x, float* __restrict__ y,
                                    int N, int C, int H, int W, int OH, int OW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * OH * OW;
    if (idx >= total) return;

    int ow = idx % OW;
    int oh = (idx / OW) % OH;
    int c = (idx / (OW * OH)) % C;
    int n = idx / (OW * OH * C);

    int ih = oh * 2;
    int iw = ow * 2;
    int base = ((n * C + c) * H + ih) * W + iw;

    float m = x[base];
    float v1 = x[base + 1];
    float v2 = x[base + W];
    float v3 = x[base + W + 1];

    m = v1 > m ? v1 : m;
    m = v2 > m ? v2 : m;
    m = v3 > m ? v3 : m;
    y[idx] = m;
}

torch::Tensor relu_inplace_cuda(torch::Tensor x) {
    int n = x.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    relu_inplace_kernel<<<blocks, threads>>>(x.data_ptr<float>(), n);
    return x;
}

torch::Tensor maxpool2x2s2_cuda(torch::Tensor x) {
    x = x.contiguous();
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int OH = H / 2;
    int OW = W / 2;

    auto y = torch::empty({N, C, OH, OW}, x.options());
    int total = N * C * OH * OW;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    maxpool2x2s2_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), N, C, H, W, OH, OW
    );
    return y;
}
"""

cpp_sources = """
torch::Tensor relu_inplace_cuda(torch::Tensor x);
torch::Tensor maxpool2x2s2_cuda(torch::Tensor x);
"""

vgg16_custom_ops = load_inline(
    name="vgg16_custom_relu_pool_ops",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["relu_inplace_cuda", "maxpool2x2s2_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()

        self.c11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.c12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.c21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.c22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.c31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.c32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.c33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.c41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.c42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.c43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.c51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.c52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.c53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

        self.ops = vgg16_custom_ops

    def _relu(self, x):
        return self.ops.relu_inplace_cuda(x)

    def _pool(self, x):
        return self.ops.maxpool2x2s2_cuda(x)

    def forward(self, x):
        x = self._relu(self.c11(x))
        x = self._relu(self.c12(x))
        x = self._pool(x)

        x = self._relu(self.c21(x))
        x = self._relu(self.c22(x))
        x = self._pool(x)

        x = self._relu(self.c31(x))
        x = self._relu(self.c32(x))
        x = self._relu(self.c33(x))
        x = self._pool(x)

        x = self._relu(self.c41(x))
        x = self._relu(self.c42(x))
        x = self._relu(self.c43(x))
        x = self._pool(x)

        x = self._relu(self.c51(x))
        x = self._relu(self.c52(x))
        x = self._relu(self.c53(x))
        x = self._pool(x)

        x = x.reshape(x.size(0), -1)
        x = self._relu(self.fc1(x))
        x = self._relu(self.fc2(x))
        x = self.fc3(x)
        return x