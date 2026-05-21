import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float sigmoidf_fast(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ float convt3d_value(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    int n, int co, int od, int oh, int ow,
    int N, int Cin, int D, int H, int W,
    int Cout, int KD, int KH, int KW,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w
) {
    float acc = b[co];

    for (int ci = 0; ci < Cin; ++ci) {
        for (int kd = 0; kd < KD; ++kd) {
            int td = od + pad_d - kd;
            if (td % stride_d != 0) continue;
            int id = td / stride_d;
            if (id < 0 || id >= D) continue;

            for (int kh = 0; kh < KH; ++kh) {
                int th = oh + pad_h - kh;
                if (th % stride_h != 0) continue;
                int ih = th / stride_h;
                if (ih < 0 || ih >= H) continue;

                for (int kw = 0; kw < KW; ++kw) {
                    int tw = ow + pad_w - kw;
                    if (tw % stride_w != 0) continue;
                    int iw = tw / stride_w;
                    if (iw < 0 || iw >= W) continue;

                    int x_idx = (((n * Cin + ci) * D + id) * H + ih) * W + iw;
                    int w_idx = ((((ci * Cout + co) * KD + kd) * KH + kh) * KW + kw);
                    acc += x[x_idx] * w[w_idx];
                }
            }
        }
    }
    return acc;
}

__global__ void fused_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    const float* __restrict__ sub,
    float* __restrict__ out,
    int N, int Cin, int D, int H, int W,
    int Cout, int KD, int KH, int KW,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int outpad_d, int outpad_h, int outpad_w,
    int pool_kd, int pool_kh, int pool_kw,
    int pool_sd, int pool_sh, int pool_sw,
    int pool_pd, int pool_ph, int pool_pw
) {
    int OD = (D - 1) * stride_d - 2 * pad_d + KD + outpad_d;
    int OH = (H - 1) * stride_h - 2 * pad_h + KH + outpad_h;
    int OW = (W - 1) * stride_w - 2 * pad_w + KW + outpad_w;

    int PD = (OD + 2 * pool_pd - pool_kd) / pool_sd + 1;
    int PH = (OH + 2 * pool_ph - pool_kh) / pool_sh + 1;
    int PW = (OW + 2 * pool_pw - pool_kw) / pool_sw + 1;

    int total = N * PD * PH * PW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int tmp = idx;
    int pw = tmp % PW; tmp /= PW;
    int ph = tmp % PH; tmp /= PH;
    int pd = tmp % PD; tmp /= PD;
    int n = tmp;

    float pooled[16];

    for (int co = 0; co < Cout; ++co) {
        float mv = -3.4028234663852886e38f;

        for (int rd = 0; rd < pool_kd; ++rd) {
            int od = pd * pool_sd + rd - pool_pd;
            if (od < 0 || od >= OD) continue;

            for (int rh = 0; rh < pool_kh; ++rh) {
                int oh = ph * pool_sh + rh - pool_ph;
                if (oh < 0 || oh >= OH) continue;

                for (int rw = 0; rw < pool_kw; ++rw) {
                    int ow = pw * pool_sw + rw - pool_pw;
                    if (ow < 0 || ow >= OW) continue;

                    float v = convt3d_value(
                        x, w, b, n, co, od, oh, ow,
                        N, Cin, D, H, W, Cout, KD, KH, KW,
                        stride_d, stride_h, stride_w,
                        pad_d, pad_h, pad_w
                    );
                    mv = fmaxf(mv, v);
                }
            }
        }
        pooled[co] = mv;
    }

    float maxv = pooled[0];
    for (int c = 1; c < Cout; ++c) maxv = fmaxf(maxv, pooled[c]);

    float sum = 0.0f;
    for (int c = 0; c < Cout; ++c) sum += expf(pooled[c] - maxv);

    float best = -3.4028234663852886e38f;
    for (int c = 0; c < Cout; ++c) {
        float y = expf(pooled[c] - maxv) / sum;
        y = y - sub[c];
        y = y * sigmoidf_fast(y);
        best = fmaxf(best, y);
    }

    out[idx] = best;
}

torch::Tensor fused_forward_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    torch::Tensor sub,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int outpad_d, int outpad_h, int outpad_w,
    int pool_kd, int pool_kh, int pool_kw,
    int pool_sd, int pool_sh, int pool_sw,
    int pool_pd, int pool_ph, int pool_pw
) {
    int N = x.size(0);
    int Cin = x.size(1);
    int D = x.size(2);
    int H = x.size(3);
    int W = x.size(4);

    int Cout = w.size(1);
    int KD = w.size(2);
    int KH = w.size(3);
    int KW = w.size(4);

    int OD = (D - 1) * stride_d - 2 * pad_d + KD + outpad_d;
    int OH = (H - 1) * stride_h - 2 * pad_h + KH + outpad_h;
    int OW = (W - 1) * stride_w - 2 * pad_w + KW + outpad_w;

    int PD = (OD + 2 * pool_pd - pool_kd) / pool_sd + 1;
    int PH = (OH + 2 * pool_ph - pool_kh) / pool_sh + 1;
    int PW = (OW + 2 * pool_pw - pool_kw) / pool_sw + 1;

    auto out = torch::empty({N, PD, PH, PW}, x.options());

    int total = N * PD * PH * PW;
    int threads = 128;
    int blocks = (total + threads - 1) / threads;

    fused_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        b.data_ptr<float>(),
        sub.data_ptr<float>(),
        out.data_ptr<float>(),
        N, Cin, D, H, W, Cout, KD, KH, KW,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        outpad_d, outpad_h, outpad_w,
        pool_kd, pool_kh, pool_kw,
        pool_sd, pool_sh, pool_sw,
        pool_pd, pool_ph, pool_pw
    );

    return out;
}
"""

cpp_sources = r"""
torch::Tensor fused_forward_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    torch::Tensor sub,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int outpad_d, int outpad_h, int outpad_w,
    int pool_kd, int pool_kh, int pool_kw,
    int pool_sd, int pool_sh, int pool_sw,
    int pool_pd, int pool_ph, int pool_pw
);
"""

fused_ops = load_inline(
    name="kb_fused_convt_pool_softmax_swish_max_v1",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["fused_forward_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


def _triple(v):
    if isinstance(v, tuple):
        return v
    return (v, v, v)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.subtract = nn.Parameter(torch.randn(out_channels))

        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.output_padding = _triple(output_padding)
        self.pool_kernel_size = _triple(pool_kernel_size)
        self.pool_stride = _triple(pool_stride)
        self.pool_padding = _triple(pool_padding)
        self.fused_ops = fused_ops

    def forward(self, x):
        return self.fused_ops.fused_forward_cuda(
            x.contiguous(),
            self.conv_transpose.weight.contiguous(),
            self.conv_transpose.bias.contiguous(),
            self.subtract.contiguous(),
            self.stride[0], self.stride[1], self.stride[2],
            self.padding[0], self.padding[1], self.padding[2],
            self.output_padding[0], self.output_padding[1], self.output_padding[2],
            self.pool_kernel_size[0], self.pool_kernel_size[1], self.pool_kernel_size[2],
            self.pool_stride[0], self.pool_stride[1], self.pool_stride[2],
            self.pool_padding[0], self.pool_padding[1], self.pool_padding[2],
        )