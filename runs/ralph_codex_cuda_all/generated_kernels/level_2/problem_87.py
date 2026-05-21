import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float mish_f32(float x) {
    float sp = x > 20.0f ? x : log1pf(expf(x));
    return x * tanhf(sp);
}

__global__ void conv2d_sub_mish_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    int N, int C, int H, int W,
    int OC, int K,
    int OH, int OW,
    float sub_total
) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int noz = blockIdx.z;
    int n = noz / OC;
    int oc = noz - n * OC;

    if (n >= N || oc >= OC || oh >= OH || ow >= OW) {
        return;
    }

    float acc = b[oc];

    int x_base_n = n * C * H * W;
    int w_base_oc = oc * C * K * K;

    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        if (c < C) {
            int x_base = x_base_n + c * H * W + oh * W + ow;
            int w_base = w_base_oc + c * K * K;

            for (int kh = 0; kh < K; ++kh) {
                int x_row = x_base + kh * W;
                int w_row = w_base + kh * K;

                for (int kw = 0; kw < K; ++kw) {
                    acc = fmaf(__ldg(x + x_row + kw), __ldg(w + w_row + kw), acc);
                }
            }
        }
    }

    acc -= sub_total;
    out[((n * OC + oc) * OH + oh) * OW + ow] = mish_f32(acc);
}

torch::Tensor conv2d_sub_mish_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    float subtract_value_1,
    float subtract_value_2
) {
    int N = (int)x.size(0);
    int C = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);
    int OC = (int)weight.size(0);
    int K = (int)weight.size(2);
    int OH = H - K + 1;
    int OW = W - K + 1;

    auto out = torch::empty({N, OC, OH, OW}, x.options());

    dim3 block(16, 16);
    dim3 grid((OW + block.x - 1) / block.x, (OH + block.y - 1) / block.y, N * OC);

    conv2d_sub_mish_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W, OC, K, OH, OW,
        subtract_value_1 + subtract_value_2
    );

    return out;
}
"""

cpp_sources = r"""
torch::Tensor conv2d_sub_mish_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    float subtract_value_1,
    float subtract_value_2
);
"""

conv2d_sub_mish_ext = load_inline(
    name="conv2d_sub_mish_ext",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["conv2d_sub_mish_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = float(subtract_value_1)
        self.subtract_value_2 = float(subtract_value_2)

    def forward(self, x):
        return conv2d_sub_mish_ext.conv2d_sub_mish_cuda(
            x.contiguous(),
            self.conv.weight.contiguous(),
            self.conv.bias.contiguous(),
            self.subtract_value_1,
            self.subtract_value_2,
        )