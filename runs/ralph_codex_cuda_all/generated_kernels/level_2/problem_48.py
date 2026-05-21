import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void conv3d_fused_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ conv_b,
    const float* __restrict__ scale,
    const float* __restrict__ post_b,
    float* __restrict__ out,
    int N, int C, int D, int H, int Wd,
    int O, int KD, int KH, int KW,
    int Do, int Ho, int Wo,
    long total
) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int ow = idx % Wo;
    long t = idx / Wo;
    int oh = t % Ho;
    t /= Ho;
    int od = t % Do;
    t /= Do;
    int oc = t % O;
    int n = t / O;

    float acc = conv_b ? __ldg(conv_b + oc) : 0.0f;

    for (int ic = 0; ic < C; ++ic) {
        for (int kz = 0; kz < KD; ++kz) {
            int id = od + kz;
            for (int ky = 0; ky < KH; ++ky) {
                int ih = oh + ky;
                const float* x_base = x + (((n * C + ic) * D + id) * H + ih) * Wd + ow;
                const float* w_base = w + (((oc * C + ic) * KD + kz) * KH + ky) * KW;
                #pragma unroll
                for (int kx = 0; kx < KW; ++kx) {
                    acc += __ldg(x_base + kx) * __ldg(w_base + kx);
                }
            }
        }
    }

    float y = acc * __ldg(scale + oc);
    y = tanhf(y);
    y = y * __ldg(post_b + oc);
    out[idx] = 1.0f / (1.0f + expf(-y));
}

torch::Tensor conv3d_fused_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor conv_b,
    torch::Tensor scale,
    torch::Tensor post_b
) {
    int N = (int)x.size(0);
    int C = (int)x.size(1);
    int D = (int)x.size(2);
    int H = (int)x.size(3);
    int Wd = (int)x.size(4);

    int O = (int)w.size(0);
    int KD = (int)w.size(2);
    int KH = (int)w.size(3);
    int KW = (int)w.size(4);

    int Do = D - KD + 1;
    int Ho = H - KH + 1;
    int Wo = Wd - KW + 1;

    auto out = torch::empty({N, O, Do, Ho, Wo}, x.options());
    long total = out.numel();

    const int threads = 256;
    const int blocks = (int)((total + threads - 1) / threads);

    conv3d_fused_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        conv_b.defined() && conv_b.numel() > 0 ? conv_b.data_ptr<float>() : nullptr,
        scale.data_ptr<float>(),
        post_b.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, D, H, Wd, O, KD, KH, KW, Do, Ho, Wo, total
    );

    return out;
}
"""

cpp_sources = r"""
torch::Tensor conv3d_fused_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor conv_b,
    torch::Tensor scale,
    torch::Tensor post_b
);
"""

_conv3d_fused = load_inline(
    name="conv3d_fused_tanh_sigmoid_fp32",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["conv3d_fused_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self._op = _conv3d_fused

    def forward(self, x):
        return self._op.conv3d_fused_cuda(
            x,
            self.conv.weight,
            self.conv.bias,
            self.scaling_factor,
            self.bias,
        )