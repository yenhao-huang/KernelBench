import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void conv1x1_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y,
    int B, int C, int O, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int HW = H * W;
    int total = B * O * HW;
    if (idx >= total) return;

    int p = idx % HW;
    int o = (idx / HW) % O;
    int n = idx / (O * HW);

    float acc = b[o];
    int xbase = n * C * HW + p;
    int wbase = o * C;
    #pragma unroll 4
    for (int c = 0; c < C; ++c) {
        acc += x[xbase + c * HW] * w[wbase + c];
    }
    y[idx] = acc;
}

__global__ void conv3x3_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y,
    int B, int C, int O, int H, int Wd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int HW = H * Wd;
    int total = B * O * HW;
    if (idx >= total) return;

    int p = idx % HW;
    int ox = p % Wd;
    int oy = p / Wd;
    int o = (idx / HW) % O;
    int n = idx / (O * HW);

    float acc = b[o];
    for (int c = 0; c < C; ++c) {
        int xbase = n * C * HW + c * HW;
        int wbase = ((o * C + c) * 3) * 3;
        for (int ky = 0; ky < 3; ++ky) {
            int iy = oy + ky - 1;
            if ((unsigned)iy >= (unsigned)H) continue;
            for (int kx = 0; kx < 3; ++kx) {
                int ix = ox + kx - 1;
                if ((unsigned)ix < (unsigned)Wd) {
                    acc += x[xbase + iy * Wd + ix] * w[wbase + ky * 3 + kx];
                }
            }
        }
    }
    y[idx] = acc;
}

__global__ void conv5x5_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y,
    int B, int C, int O, int H, int Wd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int HW = H * Wd;
    int total = B * O * HW;
    if (idx >= total) return;

    int p = idx % HW;
    int ox = p % Wd;
    int oy = p / Wd;
    int o = (idx / HW) % O;
    int n = idx / (O * HW);

    float acc = b[o];
    for (int c = 0; c < C; ++c) {
        int xbase = n * C * HW + c * HW;
        int wbase = ((o * C + c) * 5) * 5;
        for (int ky = 0; ky < 5; ++ky) {
            int iy = oy + ky - 2;
            if ((unsigned)iy >= (unsigned)H) continue;
            for (int kx = 0; kx < 5; ++kx) {
                int ix = ox + kx - 2;
                if ((unsigned)ix < (unsigned)Wd) {
                    acc += x[xbase + iy * Wd + ix] * w[wbase + ky * 5 + kx];
                }
            }
        }
    }
    y[idx] = acc;
}

__global__ void maxpool3x3_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int B, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int HW = H * W;
    int total = B * C * HW;
    if (idx >= total) return;

    int p = idx % HW;
    int ox = p % W;
    int oy = p / W;
    int c = (idx / HW) % C;
    int n = idx / (C * HW);

    float m = -FLT_MAX;
    int base = n * C * HW + c * HW;
    for (int ky = 0; ky < 3; ++ky) {
        int iy = oy + ky - 1;
        if ((unsigned)iy >= (unsigned)H) continue;
        for (int kx = 0; kx < 3; ++kx) {
            int ix = ox + kx - 1;
            if ((unsigned)ix < (unsigned)W) {
                float v = x[base + iy * W + ix];
                m = v > m ? v : m;
            }
        }
    }
    y[idx] = m;
}

torch::Tensor conv1x1_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
    int B = x.size(0), C = x.size(1), H = x.size(2), Wd = x.size(3), O = w.size(0);
    auto y = torch::empty({B, O, H, Wd}, x.options());
    int total = B * O * H * Wd;
    conv1x1_kernel<<<(total + 255) / 256, 256>>>(x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), B, C, O, H, Wd);
    return y;
}

torch::Tensor conv3x3_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
    int B = x.size(0), C = x.size(1), H = x.size(2), Wd = x.size(3), O = w.size(0);
    auto y = torch::empty({B, O, H, Wd}, x.options());
    int total = B * O * H * Wd;
    conv3x3_kernel<<<(total + 255) / 256, 256>>>(x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), B, C, O, H, Wd);
    return y;
}

torch::Tensor conv5x5_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
    int B = x.size(0), C = x.size(1), H = x.size(2), Wd = x.size(3), O = w.size(0);
    auto y = torch::empty({B, O, H, Wd}, x.options());
    int total = B * O * H * Wd;
    conv5x5_kernel<<<(total + 255) / 256, 256>>>(x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), B, C, O, H, Wd);
    return y;
}

torch::Tensor maxpool3x3_cuda(torch::Tensor x) {
    int B = x.size(0), C = x.size(1), H = x.size(2), Wd = x.size(3);
    auto y = torch::empty_like(x);
    int total = B * C * H * Wd;
    maxpool3x3_kernel<<<(total + 255) / 256, 256>>>(x.data_ptr<float>(), y.data_ptr<float>(), B, C, H, Wd);
    return y;
}
"""

cpp_sources = """
torch::Tensor conv1x1_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b);
torch::Tensor conv3x3_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b);
torch::Tensor conv5x5_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b);
torch::Tensor maxpool3x3_cuda(torch::Tensor x);
"""

inception_ops = load_inline(
    name="inception_custom_ops",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["conv1x1_cuda", "conv3x3_cuda", "conv5x5_cuda", "maxpool3x3_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        super(ModelNew, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3x3, kernel_size=1),
            nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1),
        )
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5x5, kernel_size=1),
            nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2),
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
        )
        self.ops = inception_ops

    def forward(self, x):
        b1 = self.ops.conv1x1_cuda(x, self.branch1x1.weight, self.branch1x1.bias)

        r3 = self.ops.conv1x1_cuda(x, self.branch3x3[0].weight, self.branch3x3[0].bias)
        b3 = self.ops.conv3x3_cuda(r3, self.branch3x3[1].weight, self.branch3x3[1].bias)

        r5 = self.ops.conv1x1_cuda(x, self.branch5x5[0].weight, self.branch5x5[0].bias)
        b5 = self.ops.conv5x5_cuda(r5, self.branch5x5[1].weight, self.branch5x5[1].bias)

        p = self.ops.maxpool3x3_cuda(x)
        bp = self.ops.conv1x1_cuda(p, self.branch_pool[1].weight, self.branch_pool[1].bias)

        return torch.cat((b1, b3, b5, bp), 1)