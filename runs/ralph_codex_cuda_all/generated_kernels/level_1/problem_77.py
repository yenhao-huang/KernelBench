import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void convt3d_zero_kernel(float* __restrict__ y, long total) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) y[idx] = 0.0f;
}

__global__ void convt3d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y,
    int N, int CI, int DI, int HI, int WI,
    int CO, int K, int stride, int padding, int dilation,
    int DO, int HO, int WO,
    int has_bias
) {
    long linear = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)N * CO * DO * HO * WO;
    if (linear >= total) return;

    int ow = linear % WO;
    long t = linear / WO;
    int oh = t % HO;
    t /= HO;
    int od = t % DO;
    t /= DO;
    int oc = t % CO;
    int n = t / CO;

    float acc = has_bias ? b[oc] : 0.0f;

    for (int ic = 0; ic < CI; ++ic) {
        for (int kd = 0; kd < K; ++kd) {
            int zd = od + padding - kd * dilation;
            if (zd % stride != 0) continue;
            int id = zd / stride;
            if ((unsigned)id >= (unsigned)DI) continue;

            for (int kh = 0; kh < K; ++kh) {
                int zh = oh + padding - kh * dilation;
                if (zh % stride != 0) continue;
                int ih = zh / stride;
                if ((unsigned)ih >= (unsigned)HI) continue;

                for (int kw = 0; kw < K; ++kw) {
                    int zw = ow + padding - kw * dilation;
                    if (zw % stride != 0) continue;
                    int iw = zw / stride;
                    if ((unsigned)iw >= (unsigned)WI) continue;

                    long x_idx = ((((long)n * CI + ic) * DI + id) * HI + ih) * WI + iw;
                    long w_idx = ((((long)ic * CO + oc) * K + kd) * K + kh) * K + kw;
                    acc += x[x_idx] * w[w_idx];
                }
            }
        }
    }

    y[linear] = acc;
}

torch::Tensor convt3d_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    bool has_bias
) {
    int N = x.size(0);
    int CI = x.size(1);
    int DI = x.size(2);
    int HI = x.size(3);
    int WI = x.size(4);
    int CO = weight.size(1);
    int K = weight.size(2);

    int DO = (DI - 1) * stride - 2 * padding + dilation * (K - 1) + 1;
    int HO = (HI - 1) * stride - 2 * padding + dilation * (K - 1) + 1;
    int WO = (WI - 1) * stride - 2 * padding + dilation * (K - 1) + 1;

    auto y = torch::empty({N, CO, DO, HO, WO}, x.options());

    long total = (long)N * CO * DO * HO * WO;
    const int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);

    convt3d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        has_bias ? bias.data_ptr<float>() : nullptr,
        y.data_ptr<float>(),
        N, CI, DI, HI, WI, CO, K, stride, padding, dilation, DO, HO, WO,
        has_bias ? 1 : 0
    );

    return y;
}
"""

cpp_sources = r"""
torch::Tensor convt3d_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    bool has_bias
);
"""

convt3d_ext = load_inline(
    name="convt3d_inline_fp32",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["convt3d_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation = int(dilation)
        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels, kernel_size, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in = in_channels * kernel_size * kernel_size * kernel_size
            bound = fan_in ** -0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias_tensor = self.bias if self.bias is not None else torch.empty(0, device=x.device, dtype=x.dtype)
        return convt3d_ext.convt3d_cuda(
            x.contiguous(),
            self.weight.contiguous(),
            bias_tensor,
            self.stride,
            self.padding,
            self.dilation,
            self.bias is not None,
        )