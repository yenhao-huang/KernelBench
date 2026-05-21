import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

maxpool2d_cpp_source = """
torch::Tensor maxpool2d_cuda(torch::Tensor x, int kernel_size, int stride, int padding, int dilation);
"""

maxpool2d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void maxpool2d_k4s1p1_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int total,
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
    int nc = t / OH;

    int base = nc * H * W;
    int ih0 = oh - 1;
    int iw0 = ow - 1;

    float m = -FLT_MAX;

    #pragma unroll
    for (int kh = 0; kh < 4; ++kh) {
        int ih = ih0 + kh;
        if ((unsigned)ih < (unsigned)H) {
            int row = base + ih * W;
            #pragma unroll
            for (int kw = 0; kw < 4; ++kw) {
                int iw = iw0 + kw;
                if ((unsigned)iw < (unsigned)W) {
                    float v = __ldg(x + row + iw);
                    m = fmaxf(m, v);
                }
            }
        }
    }

    y[idx] = m;
}

__global__ void maxpool2d_generic_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int total,
    int H,
    int W,
    int OH,
    int OW,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int ow = idx % OW;
    int t = idx / OW;
    int oh = t % OH;
    int nc = t / OH;

    int base = nc * H * W;
    int ih0 = oh * stride - padding;
    int iw0 = ow * stride - padding;

    float m = -FLT_MAX;

    for (int kh = 0; kh < kernel_size; ++kh) {
        int ih = ih0 + kh * dilation;
        if ((unsigned)ih < (unsigned)H) {
            int row = base + ih * W;
            for (int kw = 0; kw < kernel_size; ++kw) {
                int iw = iw0 + kw * dilation;
                if ((unsigned)iw < (unsigned)W) {
                    float v = __ldg(x + row + iw);
                    m = fmaxf(m, v);
                }
            }
        }
    }

    y[idx] = m;
}

torch::Tensor maxpool2d_cuda(torch::Tensor x, int kernel_size, int stride, int padding, int dilation) {
    int N = (int)x.size(0);
    int C = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);

    int effective = dilation * (kernel_size - 1) + 1;
    int OH = (H + 2 * padding - effective) / stride + 1;
    int OW = (W + 2 * padding - effective) / stride + 1;

    auto y = torch::empty({N, C, OH, OW}, x.options());
    int total = N * C * OH * OW;

    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    if (kernel_size == 4 && stride == 1 && padding == 1 && dilation == 1) {
        maxpool2d_k4s1p1_kernel<<<blocks, threads>>>(
            x.data_ptr<float>(), y.data_ptr<float>(), total, H, W, OH, OW
        );
    } else {
        maxpool2d_generic_kernel<<<blocks, threads>>>(
            x.data_ptr<float>(), y.data_ptr<float>(), total, H, W, OH, OW,
            kernel_size, stride, padding, dilation
        );
    }

    return y;
}
"""

maxpool2d_ext = load_inline(
    name="maxpool2d_inline_ext",
    cpp_sources=maxpool2d_cpp_source,
    cuda_sources=maxpool2d_cuda_source,
    functions=["maxpool2d_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super(ModelNew, self).__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation = int(dilation)
        self.maxpool2d_ext = maxpool2d_ext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool2d_ext.maxpool2d_cuda(
            x, self.kernel_size, self.stride, self.padding, self.dilation
        )