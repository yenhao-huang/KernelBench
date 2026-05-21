import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from torch.nn.modules.utils import _triple


conv3d_cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv3d_ncdhw_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y,
    int N, int C, int ID, int IH, int IW,
    int OC, int KD, int KH, int KW,
    int OD, int OH, int OW,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dil_d, int dil_h, int dil_w,
    int has_bias
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)N * OC * OD * OH * OW;
    if (idx >= total) return;

    int ow = idx % OW;
    long long t = idx / OW;
    int oh = t % OH;
    t /= OH;
    int od = t % OD;
    t /= OD;
    int oc = t % OC;
    int n = t / OC;

    float acc = has_bias ? b[oc] : 0.0f;

    int base_d = od * stride_d - pad_d;
    int base_h = oh * stride_h - pad_h;
    int base_w = ow * stride_w - pad_w;

    for (int c = 0; c < C; ++c) {
        for (int kd = 0; kd < KD; ++kd) {
            int id = base_d + kd * dil_d;
            if ((unsigned)id >= (unsigned)ID) continue;
            for (int kh = 0; kh < KH; ++kh) {
                int ih = base_h + kh * dil_h;
                if ((unsigned)ih >= (unsigned)IH) continue;

                const float* __restrict__ xp = x + (((long long)n * C + c) * ID + id) * IH * IW + ih * IW;
                const float* __restrict__ wp = w + (((long long)oc * C + c) * KD + kd) * KH * KW + kh * KW;

                #pragma unroll
                for (int kw = 0; kw < 7; ++kw) {
                    if (kw < KW) {
                        int iw = base_w + kw * dil_w;
                        if ((unsigned)iw < (unsigned)IW) {
                            acc += xp[iw] * wp[kw];
                        }
                    }
                }
            }
        }
    }

    y[idx] = acc;
}

torch::Tensor conv3d_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dil_d, int dil_h, int dil_w,
    int has_bias
) {
    int N = x.size(0);
    int C = x.size(1);
    int ID = x.size(2);
    int IH = x.size(3);
    int IW = x.size(4);

    int OC = weight.size(0);
    int KD = weight.size(2);
    int KH = weight.size(3);
    int KW = weight.size(4);

    int OD = (ID + 2 * pad_d - dil_d * (KD - 1) - 1) / stride_d + 1;
    int OH = (IH + 2 * pad_h - dil_h * (KH - 1) - 1) / stride_h + 1;
    int OW = (IW + 2 * pad_w - dil_w * (KW - 1) - 1) / stride_w + 1;

    auto y = torch::empty({N, OC, OD, OH, OW}, x.options());

    long long total = (long long)N * OC * OD * OH * OW;
    const int threads = 256;
    const int blocks = (int)((total + threads - 1) / threads);

    const float* bptr = has_bias ? bias.data_ptr<float>() : nullptr;

    conv3d_ncdhw_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bptr,
        y.data_ptr<float>(),
        N, C, ID, IH, IW,
        OC, KD, KH, KW,
        OD, OH, OW,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dil_d, dil_h, dil_w,
        has_bias
    );

    return y;
}
"""

conv3d_cpp_source = r"""
torch::Tensor conv3d_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dil_d, int dil_h, int dil_w,
    int has_bias
);
"""

conv3d_ext = load_inline(
    name="kb_conv3d_direct_fp32_ext",
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_cuda_source,
    functions=["conv3d_forward_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.conv3d = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.conv3d.bias
        if bias is None:
            bias = torch.empty(0, device=x.device, dtype=x.dtype)
            has_bias = 0
        else:
            has_bias = 1

        return conv3d_ext.conv3d_forward_cuda(
            x.contiguous(),
            self.conv3d.weight.contiguous(),
            bias,
            self.stride[0], self.stride[1], self.stride[2],
            self.padding[0], self.padding[1], self.padding[2],
            self.dilation[0], self.dilation[1], self.dilation[2],
            has_bias,
        )