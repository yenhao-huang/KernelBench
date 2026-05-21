import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

depthwise_conv2d_cpp_source = """
torch::Tensor depthwise_conv2d_cuda(torch::Tensor x, torch::Tensor w, int stride, int padding);
"""

depthwise_conv2d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void depthwise3x3_s1p0_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ y,
    int total,
    int N,
    int C,
    int H,
    int W,
    int OH,
    int OW
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int ow = idx % OW;
    int t = idx / OW;
    int oh = t % OH;
    t /= OH;
    int c = t % C;
    int n = t / C;

    int x_base = ((n * C + c) * H + oh) * W + ow;
    int w_base = c * 9;

    float v = 0.0f;
    v += x[x_base] * w[w_base];
    v += x[x_base + 1] * w[w_base + 1];
    v += x[x_base + 2] * w[w_base + 2];

    x_base += W;
    v += x[x_base] * w[w_base + 3];
    v += x[x_base + 1] * w[w_base + 4];
    v += x[x_base + 2] * w[w_base + 5];

    x_base += W;
    v += x[x_base] * w[w_base + 6];
    v += x[x_base + 1] * w[w_base + 7];
    v += x[x_base + 2] * w[w_base + 8];

    y[idx] = v;
}

__global__ void depthwise_generic_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ y,
    int total,
    int C,
    int H,
    int W,
    int K,
    int stride,
    int padding,
    int OH,
    int OW
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int ow = idx % OW;
    int t = idx / OW;
    int oh = t % OH;
    t /= OH;
    int c = t % C;
    int n = t / C;

    float acc = 0.0f;
    int ih0 = oh * stride - padding;
    int iw0 = ow * stride - padding;

    for (int kh = 0; kh < K; ++kh) {
        int ih = ih0 + kh;
        if ((unsigned)ih >= (unsigned)H) continue;
        for (int kw = 0; kw < K; ++kw) {
            int iw = iw0 + kw;
            if ((unsigned)iw < (unsigned)W) {
                acc += x[((n * C + c) * H + ih) * W + iw] * w[(c * K + kh) * K + kw];
            }
        }
    }

    y[idx] = acc;
}

torch::Tensor depthwise_conv2d_cuda(torch::Tensor x, torch::Tensor w, int stride, int padding) {
    int N = (int)x.size(0);
    int C = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);
    int K = (int)w.size(2);

    int OH = (H + 2 * padding - K) / stride + 1;
    int OW = (W + 2 * padding - K) / stride + 1;

    auto y = torch::empty({N, C, OH, OW}, x.options());
    int total = N * C * OH * OW;

    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    if (K == 3 && stride == 1 && padding == 0) {
        depthwise3x3_s1p0_kernel<<<blocks, threads>>>(
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            y.data_ptr<float>(),
            total, N, C, H, W, OH, OW
        );
    } else {
        depthwise_generic_kernel<<<blocks, threads>>>(
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            y.data_ptr<float>(),
            total, C, H, W, K, stride, padding, OH, OW
        );
    }

    return y;
}
"""

_depthwise_conv2d = load_inline(
    name="depthwise_conv2d_kernelbench_fp32",
    cpp_sources=depthwise_conv2d_cpp_source,
    cuda_sources=depthwise_conv2d_cuda_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _depthwise_conv2d.depthwise_conv2d_cuda(
            x.contiguous(),
            self.conv2d.weight.contiguous(),
            self.stride,
            self.padding,
        )