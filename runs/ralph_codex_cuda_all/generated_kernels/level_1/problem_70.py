import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cpp_sources = """
torch::Tensor conv_transpose3d_fp32_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t dilation,
    int64_t groups);
"""

cuda_sources = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose3d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ y,
    long total,
    int N, int Cin, int Din, int Hin, int Win,
    int Cout, int Dout, int Hout, int Wout,
    int K, int stride, int padding, int dilation, int groups,
    int has_bias) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int ow = idx % Wout;
    long t = idx / Wout;
    int oh = t % Hout;
    t /= Hout;
    int od = t % Dout;
    t /= Dout;
    int oc = t % Cout;
    int n = t / Cout;

    int cout_per_group = Cout / groups;
    int cin_per_group = Cin / groups;
    int g = oc / cout_per_group;
    int ic_start = g * cin_per_group;
    int ic_end = ic_start + cin_per_group;
    int ocg = oc - g * cout_per_group;

    float acc = has_bias ? bias[oc] : 0.0f;

    for (int ic = ic_start; ic < ic_end; ++ic) {
        for (int kd = 0; kd < K; ++kd) {
            int zd = od + padding - kd * dilation;
            if (zd < 0 || zd % stride != 0) continue;
            int id = zd / stride;
            if (id < 0 || id >= Din) continue;

            for (int kh = 0; kh < K; ++kh) {
                int zh = oh + padding - kh * dilation;
                if (zh < 0 || zh % stride != 0) continue;
                int ih = zh / stride;
                if (ih < 0 || ih >= Hin) continue;

                const float* xrow = x + (((((long)n * Cin + ic) * Din + id) * Hin + ih) * Win);
                const float* wbase = w + (((((long)ic * cout_per_group + ocg) * K + kd) * K + kh) * K);

                #pragma unroll
                for (int kw = 0; kw < 3; ++kw) {
                    if (kw >= K) break;
                    int zw = ow + padding - kw * dilation;
                    if (zw < 0 || zw % stride != 0) continue;
                    int iw = zw / stride;
                    if (iw >= 0 && iw < Win) {
                        acc += xrow[iw] * wbase[kw];
                    }
                }
            }
        }
    }

    y[idx] = acc;
}

torch::Tensor conv_transpose3d_fp32_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int64_t stride64,
    int64_t padding64,
    int64_t output_padding64,
    int64_t dilation64,
    int64_t groups64) {
    int N = (int)x.size(0);
    int Cin = (int)x.size(1);
    int Din = (int)x.size(2);
    int Hin = (int)x.size(3);
    int Win = (int)x.size(4);

    int K = (int)weight.size(2);
    int groups = (int)groups64;
    int Cout = (int)weight.size(1) * groups;

    int stride = (int)stride64;
    int padding = (int)padding64;
    int output_padding = (int)output_padding64;
    int dilation = (int)dilation64;

    int Dout = (Din - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1;
    int Hout = (Hin - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1;
    int Wout = (Win - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1;

    auto y = torch::empty({N, Cout, Dout, Hout, Wout}, x.options());

    const float* bias_ptr = nullptr;
    int has_bias = 0;
    if (bias.has_value() && bias.value().defined()) {
        bias_ptr = bias.value().data_ptr<float>();
        has_bias = 1;
    }

    long total = (long)N * Cout * Dout * Hout * Wout;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);

    conv_transpose3d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        total,
        N, Cin, Din, Hin, Win,
        Cout, Dout, Hout, Wout,
        K, stride, padding, dilation, groups,
        has_bias);

    return y;
}
"""

conv_transpose3d_ext = load_inline(
    name="conv_transpose3d_fp32_inline_ext",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["conv_transpose3d_fp32_cuda"],
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
        output_padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            (kernel_size, kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.stride = stride if isinstance(stride, int) else stride[0]
        self.padding = padding if isinstance(padding, int) else padding[0]
        self.output_padding = output_padding if isinstance(output_padding, int) else output_padding[0]
        self.dilation = dilation if isinstance(dilation, int) else dilation[0]
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv_transpose3d_ext.conv_transpose3d_fp32_cuda(
            x.contiguous(),
            self.conv_transpose3d.weight.contiguous(),
            self.conv_transpose3d.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
            self.groups,
        )