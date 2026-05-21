import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

depthwise_kx1_cpp = """
torch::Tensor depthwise_kx1_forward_cuda(torch::Tensor x, torch::Tensor weight, c10::optional<torch::Tensor> bias, int stride, int padding, int dilation);
"""

depthwise_kx1_cuda = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void depthwise_k3x1_s1p0_vec4_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    int N, int C, int H, int W
) {
    int vec_col = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y;
    int nc = blockIdx.z;
    int n = nc / C;
    int c = nc - n * C;
    int ow = vec_col * 4;

    if (ow + 3 >= W) return;

    int Hout = H - 2;
    const float* x_base = x + ((n * C + c) * H + oh) * W + ow;
    float* o_base = out + ((n * C + c) * Hout + oh) * W + ow;

    float k0 = w[c * 3 + 0];
    float k1 = w[c * 3 + 1];
    float k2 = w[c * 3 + 2];
    float bv = b ? b[c] : 0.0f;

    float4 r0 = *reinterpret_cast<const float4*>(x_base);
    float4 r1 = *reinterpret_cast<const float4*>(x_base + W);
    float4 r2 = *reinterpret_cast<const float4*>(x_base + 2 * W);

    float4 y;
    y.x = r0.x * k0 + r1.x * k1 + r2.x * k2 + bv;
    y.y = r0.y * k0 + r1.y * k1 + r2.y * k2 + bv;
    y.z = r0.z * k0 + r1.z * k1 + r2.z * k2 + bv;
    y.w = r0.w * k0 + r1.w * k1 + r2.w * k2 + bv;

    *reinterpret_cast<float4*>(o_base) = y;
}

__global__ void depthwise_kx1_generic_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    int total,
    int N, int C, int H, int W,
    int K, int stride, int padding, int dilation,
    int Hout, int Wout
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int ow = idx % Wout;
    int t = idx / Wout;
    int oh = t % Hout;
    t /= Hout;
    int c = t % C;
    int n = t / C;

    int ih0 = oh * stride - padding;
    int iw = ow * stride - padding;

    float acc = b ? b[c] : 0.0f;

    if (iw >= 0 && iw < W) {
        const float* x_chan = x + ((n * C + c) * H) * W;
        const float* w_chan = w + c * K;
        #pragma unroll
        for (int kh = 0; kh < 16; ++kh) {
            if (kh >= K) break;
            int ih = ih0 + kh * dilation;
            if (ih >= 0 && ih < H) {
                acc += x_chan[ih * W + iw] * w_chan[kh];
            }
        }
    }

    out[idx] = acc;
}

torch::Tensor depthwise_kx1_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation
) {
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int K = weight.size(2);

    int Hout = (H + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    int Wout = (W + 2 * padding - 1) / stride + 1;

    auto out = torch::empty({N, C, Hout, Wout}, x.options());
    const float* bptr = bias.has_value() && bias.value().defined() ? bias.value().data_ptr<float>() : nullptr;

    if (K == 3 && stride == 1 && padding == 0 && dilation == 1 && (W % 4 == 0)) {
        dim3 block(128);
        dim3 grid((W / 4 + block.x - 1) / block.x, Hout, N * C);
        depthwise_k3x1_s1p0_vec4_kernel<<<grid, block>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bptr,
            out.data_ptr<float>(),
            N, C, H, W
        );
    } else {
        int total = N * C * Hout * Wout;
        int block = 256;
        int grid = (total + block - 1) / block;
        depthwise_kx1_generic_kernel<<<grid, block>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bptr,
            out.data_ptr<float>(),
            total, N, C, H, W, K, stride, padding, dilation, Hout, Wout
        );
    }

    return out;
}
"""

depthwise_kx1_ext = load_inline(
    name="depthwise_kx1_ext",
    cpp_sources=depthwise_kx1_cpp,
    cuda_sources=depthwise_kx1_cuda,
    functions=["depthwise_kx1_forward_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(kernel_size, 1),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return depthwise_kx1_ext.depthwise_kx1_forward_cuda(
            x,
            self.conv2d.weight,
            self.conv2d.bias,
            self.stride,
            self.padding,
            self.dilation,
        )