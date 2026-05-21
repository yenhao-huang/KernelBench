import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void deconv_bn_pool4_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    const float* __restrict__ bn_w,
    const float* __restrict__ bn_b,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float* __restrict__ out,
    int N, int Cin, int Din, int Hin, int Win,
    int Cout, int K, int stride, int padding,
    int Dout, int Hout, int Wout,
    int PoutD, int PoutH, int PoutW,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * Cout * PoutD * PoutH * PoutW;
    if (idx >= total) return;

    int ow = idx % PoutW;
    int t = idx / PoutW;
    int oh = t % PoutH;
    t /= PoutH;
    int od = t % PoutD;
    t /= PoutD;
    int oc = t % Cout;
    int n = t / Cout;

    float acc_pool = 0.0f;

    #pragma unroll
    for (int pd = 0; pd < 4; ++pd) {
        int zd = od * 4 + pd;
        #pragma unroll
        for (int ph = 0; ph < 4; ++ph) {
            int yh = oh * 4 + ph;
            #pragma unroll
            for (int pw = 0; pw < 4; ++pw) {
                int xw = ow * 4 + pw;
                float v = 0.0f;

                for (int ic = 0; ic < Cin; ++ic) {
                    for (int kd = 0; kd < K; ++kd) {
                        int id_num = zd + padding - kd;
                        if (id_num < 0 || id_num % stride != 0) continue;
                        int id = id_num / stride;
                        if (id < 0 || id >= Din) continue;

                        for (int kh = 0; kh < K; ++kh) {
                            int ih_num = yh + padding - kh;
                            if (ih_num < 0 || ih_num % stride != 0) continue;
                            int ih = ih_num / stride;
                            if (ih < 0 || ih >= Hin) continue;

                            for (int kw = 0; kw < K; ++kw) {
                                int iw_num = xw + padding - kw;
                                if (iw_num < 0 || iw_num % stride != 0) continue;
                                int iw = iw_num / stride;
                                if (iw < 0 || iw >= Win) continue;

                                int x_idx = (((n * Cin + ic) * Din + id) * Hin + ih) * Win + iw;
                                int w_idx = ((((ic * Cout + oc) * K + kd) * K + kh) * K + kw);
                                v += x[x_idx] * w[w_idx];
                            }
                        }
                    }
                }

                v += bias[oc];
                acc_pool += v;
            }
        }
    }

    float mean = running_mean[oc];
    float inv_std = rsqrtf(running_var[oc] + eps);
    float scale = bn_w[oc] * inv_std;
    float shift = bn_b[oc] - mean * scale;
    out[idx] = acc_pool * (scale * 0.015625f) + shift;
}

torch::Tensor deconv_bn_pool4_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor bias,
    torch::Tensor bn_w,
    torch::Tensor bn_b,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    int stride,
    int padding,
    double eps
) {
    int N = x.size(0);
    int Cin = x.size(1);
    int Din = x.size(2);
    int Hin = x.size(3);
    int Win = x.size(4);
    int Cout = w.size(1);
    int K = w.size(2);

    int Dout = (Din - 1) * stride - 2 * padding + K;
    int Hout = (Hin - 1) * stride - 2 * padding + K;
    int Wout = (Win - 1) * stride - 2 * padding + K;

    int PoutD = Dout / 4;
    int PoutH = Hout / 4;
    int PoutW = Wout / 4;

    auto out = torch::empty({N, Cout, PoutD, PoutH, PoutW}, x.options());

    int total = N * Cout * PoutD * PoutH * PoutW;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    deconv_bn_pool4_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        bias.data_ptr<float>(),
        bn_w.data_ptr<float>(),
        bn_b.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        out.data_ptr<float>(),
        N, Cin, Din, Hin, Win,
        Cout, K, stride, padding,
        Dout, Hout, Wout,
        PoutD, PoutH, PoutW,
        (float)eps
    );

    return out;
}
"""

cpp_sources = r"""
torch::Tensor deconv_bn_pool4_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor bias,
    torch::Tensor bn_w,
    torch::Tensor bn_b,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    int stride,
    int padding,
    double eps
);
"""

_deconv_bn_pool4 = load_inline(
    name="deconv_bn_pool4_inline",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["deconv_bn_pool4_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.stride = stride
        self.padding = padding
        self.op = _deconv_bn_pool4

    def forward(self, x):
        return self.op.deconv_bn_pool4_cuda(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.stride,
            self.padding,
            self.batch_norm.eps,
        )