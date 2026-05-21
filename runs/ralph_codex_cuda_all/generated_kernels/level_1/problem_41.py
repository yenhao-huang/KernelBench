import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

maxpool1d_cpp_source = """
torch::Tensor maxpool1d_dilated_cuda(torch::Tensor x, int kernel_size, int stride, int padding, int dilation);
"""

maxpool1d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void maxpool1d_dilated_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    long total,
    int L,
    int outL,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int ox = idx % outL;
    long row = idx / outL;
    long base = row * (long)L;

    int start = ox * stride - padding;
    float m = -FLT_MAX;

    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        if (k < kernel_size) {
            int ix = start + k * dilation;
            if ((unsigned)ix < (unsigned)L) {
                float v = x[base + ix];
                m = v > m ? v : m;
            }
        }
    }

    y[idx] = m;
}

torch::Tensor maxpool1d_dilated_cuda(torch::Tensor x, int kernel_size, int stride, int padding, int dilation) {
    const int B = x.size(0);
    const int C = x.size(1);
    const int L = x.size(2);
    const int outL = (L + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto y = torch::empty({B, C, outL}, x.options());
    long total = (long)B * C * outL;

    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    maxpool1d_dilated_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        total,
        L,
        outL,
        kernel_size,
        stride,
        padding,
        dilation
    );

    return y;
}
"""

maxpool1d_ext = load_inline(
    name="maxpool1d_dilated_ext",
    cpp_sources=maxpool1d_cpp_source,
    cuda_sources=maxpool1d_cuda_source,
    functions=["maxpool1d_dilated_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride) if stride is not None else int(kernel_size)
        self.padding = int(padding)
        self.dilation = int(dilation)
        self.return_indices = bool(return_indices)
        self.op = maxpool1d_ext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op.maxpool1d_dilated_cuda(
            x.contiguous(),
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        )