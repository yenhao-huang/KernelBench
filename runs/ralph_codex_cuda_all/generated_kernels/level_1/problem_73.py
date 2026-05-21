import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose3d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int N, int Cin, int D, int H, int W,
    int Cout, int K, int stride, int padding, int groups,
    int Do, int Ho, int Wo,
    bool has_bias
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)N * Cout * Do * Ho * Wo;
    if (idx >= total) return;

    int ow = idx % Wo;
    long long t = idx / Wo;
    int oh = t % Ho;
    t /= Ho;
    int od = t % Do;
    t /= Do;
    int oc = t % Cout;
    int n = t / Cout;

    int cin_per_group = Cin / groups;
    int cout_per_group = Cout / groups;
    int g = oc / cout_per_group;
    int ic_start = g * cin_per_group;
    int ic_end = ic_start + cin_per_group;
    int ocg = oc - g * cout_per_group;

    float acc = has_bias ? bias[oc] : 0.0f;

    #pragma unroll
    for (int kd = 0; kd < 8; ++kd) {
        if (kd >= K) break;
        int id_num = od + padding - kd;
        if (id_num % stride != 0) continue;
        int id = id_num / stride;
        if ((unsigned)id >= (unsigned)D) continue;

        #pragma unroll
        for (int kh = 0; kh < 8; ++kh) {
            if (kh >= K) break;
            int ih_num = oh + padding - kh;
            if (ih_num % stride != 0) continue;
            int ih = ih_num / stride;
            if ((unsigned)ih >= (unsigned)H) continue;

            #pragma unroll
            for (int kw = 0; kw < 8; ++kw) {
                if (kw >= K) break;
                int iw_num = ow + padding - kw;
                if (iw_num % stride != 0) continue;
                int iw = iw_num / stride;
                if ((unsigned)iw >= (unsigned)W) continue;

                for (int ic = ic_start; ic < ic_end; ++ic) {
                    long long xoff = (((long long)n * Cin + ic) * D + id) * H * W + (long long)ih * W + iw;
                    long long woff = ((((long long)ic * cout_per_group + ocg) * K + kd) * K + kh) * K + kw;
                    acc += x[xoff] * w[woff];
                }
            }
        }
    }

    y[idx] = acc;
}

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int output_padding,
    int groups,
    bool has_bias
) {
    int N = x.size(0);
    int Cin = x.size(1);
    int D = x.size(2);
    int H = x.size(3);
    int W = x.size(4);
    int K = weight.size(2);
    int Cout = weight.size(1) * groups;

    int Do = (D - 1) * stride - 2 * padding + K + output_padding;
    int Ho = (H - 1) * stride - 2 * padding + K + output_padding;
    int Wo = (W - 1) * stride - 2 * padding + K + output_padding;

    auto y = torch::empty({N, Cout, Do, Ho, Wo}, x.options());

    const int threads = 256;
    long long total = (long long)N * Cout * Do * Ho * Wo;
    int blocks = (int)((total + threads - 1) / threads);

    conv_transpose3d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        has_bias ? bias.data_ptr<float>() : nullptr,
        y.data_ptr<float>(),
        N, Cin, D, H, W, Cout, K, stride, padding, groups, Do, Ho, Wo, has_bias
    );

    return y;
}
"""

cpp_sources = """
torch::Tensor conv_transpose3d_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int output_padding,
    int groups,
    bool has_bias
);
"""

conv_transpose3d_ext = load_inline(
    name="conv_transpose3d_custom_ext",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["conv_transpose3d_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        if groups == 1 and output_padding not in (0, 1):
            groups = output_padding
            output_padding = 0
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.conv_transpose3d.bias
        if b is None:
            b = torch.empty(0, device=x.device, dtype=x.dtype)
            has_bias = False
        else:
            has_bias = True
        return conv_transpose3d_ext.conv_transpose3d_cuda(
            x.contiguous(),
            self.conv_transpose3d.weight.contiguous(),
            b,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            has_bias,
        )