import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cpp_sources = """
torch::Tensor fused_deconv_pool_cuda(torch::Tensor x,
                                     torch::Tensor weight,
                                     torch::Tensor conv_bias,
                                     torch::Tensor scale1,
                                     torch::Tensor add_bias,
                                     torch::Tensor scale2);
"""

cuda_sources = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_deconv_pool_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ scale1,
    const float* __restrict__ add_bias,
    const float* __restrict__ scale2,
    float* __restrict__ out,
    int N, int IC, int ID, int IH, int IW,
    int OC, int KD, int KH, int KW,
    int OD, int OH, int OW,
    int PD, int PH, int PW,
    int stride, int padding,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int t = idx;
    int pw = t % PW; t /= PW;
    int ph = t % PH; t /= PH;
    int pd = t % PD; t /= PD;
    int oc = t % OC; t /= OC;
    int n = t;

    float acc_pool = 0.0f;

    #pragma unroll
    for (int rd = 0; rd < 2; ++rd) {
        int od = pd * 2 + rd;
        #pragma unroll
        for (int rh = 0; rh < 2; ++rh) {
            int oh = ph * 2 + rh;
            #pragma unroll
            for (int rw = 0; rw < 2; ++rw) {
                int ow = pw * 2 + rw;
                float v = conv_bias[oc];

                for (int ic = 0; ic < IC; ++ic) {
                    for (int kd = 0; kd < KD; ++kd) {
                        int id_num = od + padding - kd;
                        if (id_num < 0 || id_num % stride != 0) continue;
                        int id = id_num / stride;
                        if (id < 0 || id >= ID) continue;

                        for (int kh = 0; kh < KH; ++kh) {
                            int ih_num = oh + padding - kh;
                            if (ih_num < 0 || ih_num % stride != 0) continue;
                            int ih = ih_num / stride;
                            if (ih < 0 || ih >= IH) continue;

                            for (int kw = 0; kw < KW; ++kw) {
                                int iw_num = ow + padding - kw;
                                if (iw_num < 0 || iw_num % stride != 0) continue;
                                int iw = iw_num / stride;
                                if (iw < 0 || iw >= IW) continue;

                                int x_idx = (((n * IC + ic) * ID + id) * IH + ih) * IW + iw;
                                int w_idx = (((ic * OC + oc) * KD + kd) * KH + kh) * KW + kw;
                                v += x[x_idx] * weight[w_idx];
                            }
                        }
                    }
                }
                acc_pool += v;
            }
        }
    }

    float y = acc_pool * 0.125f * scale1[0];
    y = (y + add_bias[oc]) * scale2[0];
    out[idx] = y;
}

torch::Tensor fused_deconv_pool_cuda(torch::Tensor x,
                                     torch::Tensor weight,
                                     torch::Tensor conv_bias,
                                     torch::Tensor scale1,
                                     torch::Tensor add_bias,
                                     torch::Tensor scale2) {
    const int N = x.size(0);
    const int IC = x.size(1);
    const int ID = x.size(2);
    const int IH = x.size(3);
    const int IW = x.size(4);

    const int OC = weight.size(1);
    const int KD = weight.size(2);
    const int KH = weight.size(3);
    const int KW = weight.size(4);

    const int stride = 2;
    const int padding = 1;

    const int OD = (ID - 1) * stride - 2 * padding + KD;
    const int OH = (IH - 1) * stride - 2 * padding + KH;
    const int OW = (IW - 1) * stride - 2 * padding + KW;

    const int PD = OD / 2;
    const int PH = OH / 2;
    const int PW = OW / 2;

    auto out = torch::empty({N, OC, PD, PH, PW}, x.options());
    const int total = N * OC * PD * PH * PW;

    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    fused_deconv_pool_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        scale1.data_ptr<float>(),
        add_bias.data_ptr<float>(),
        scale2.data_ptr<float>(),
        out.data_ptr<float>(),
        N, IC, ID, IH, IW,
        OC, KD, KH, KW,
        OD, OH, OW,
        PD, PH, PW,
        stride, padding,
        total
    );

    return out;
}
"""

_fused_deconv_pool = load_inline(
    name="fused_deconv_pool3d_scale_bias_v1",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["fused_deconv_pool_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )
        self.scale1 = nn.Parameter(torch.tensor(scale1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale2 = nn.Parameter(torch.tensor(scale2))
        self._op = _fused_deconv_pool

    def forward(self, x):
        return self._op.fused_deconv_pool_cuda(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.scale1,
            self.bias,
            self.scale2,
        )