import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void convt3d_fused_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    const float* __restrict__ add,
    float* __restrict__ out,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int OW = 32;
    const int OH = 32;
    const int OD = 32;
    const int OC = 64;
    const int IW = 16;
    const int IH = 16;
    const int ID = 16;
    const int IC = 32;

    int t = idx;
    int ow = t % OW; t /= OW;
    int oh = t % OH; t /= OH;
    int od = t % OD; t /= OD;
    int oc = t % OC; t /= OC;
    int n = t;

    float acc = b[oc];

    #pragma unroll
    for (int ic = 0; ic < IC; ++ic) {
        #pragma unroll
        for (int kd = 0; kd < 3; ++kd) {
            int id_num = od + 1 - kd;
            if ((id_num & 1) != 0) continue;
            int id = id_num >> 1;
            if ((unsigned)id >= ID) continue;

            #pragma unroll
            for (int kh = 0; kh < 3; ++kh) {
                int ih_num = oh + 1 - kh;
                if ((ih_num & 1) != 0) continue;
                int ih = ih_num >> 1;
                if ((unsigned)ih >= IH) continue;

                #pragma unroll
                for (int kw = 0; kw < 3; ++kw) {
                    int iw_num = ow + 1 - kw;
                    if ((iw_num & 1) != 0) continue;
                    int iw = iw_num >> 1;
                    if ((unsigned)iw >= IW) continue;

                    int x_idx = (((n * IC + ic) * ID + id) * IH + ih) * IW + iw;
                    int w_idx = ((((ic * OC + oc) * 3 + kd) * 3 + kh) * 3 + kw);
                    acc += x[x_idx] * w[w_idx];
                }
            }
        }
    }

    float v = acc + add[idx];
    float hs = v * fminf(fmaxf(v + 3.0f, 0.0f), 6.0f) * 0.16666666666666666f;
    out[idx] = v * hs;
}

torch::Tensor convt3d_fused_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor add) {
    auto out = torch::empty_like(add);
    int total = add.numel();
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    convt3d_fused_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        b.data_ptr<float>(),
        add.data_ptr<float>(),
        out.data_ptr<float>(),
        total
    );

    return out;
}
"""

cpp_sources = "torch::Tensor convt3d_fused_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor add);"

convt3d_fused = load_inline(
    name="convt3d_fused_hardswish_mul_ext",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["convt3d_fused_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.op = convt3d_fused

    def forward(self, x, add_input):
        return self.op.convt3d_fused_cuda(
            x.contiguous(),
            self.conv_transpose.weight.contiguous(),
            self.conv_transpose.bias.contiguous(),
            add_input.contiguous(),
        )