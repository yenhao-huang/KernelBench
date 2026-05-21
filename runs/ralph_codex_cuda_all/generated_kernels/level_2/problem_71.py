import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_div_leaky_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    int N, int C, int H, int W,
    int O, int K,
    int OH, int OW,
    float divisor
) {
    extern __shared__ float smem[];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tile_w = blockDim.x + K - 1;
    const int tile_h = blockDim.y + K - 1;

    const int tile_x = blockIdx.x * blockDim.x;
    const int tile_y = blockIdx.y * blockDim.y;
    const int no = blockIdx.z;
    const int n = no / O;
    const int oc = no - n * O;

    const int shared_elems = C * tile_h * tile_w;
    const int tid = ty * blockDim.x + tx;
    const int nthreads = blockDim.x * blockDim.y;

    for (int idx = tid; idx < shared_elems; idx += nthreads) {
        int sx = idx % tile_w;
        int tmp = idx / tile_w;
        int sy = tmp % tile_h;
        int ic = tmp / tile_h;

        int gx = tile_x + sx;
        int gy = tile_y + sy;

        float v = 0.0f;
        if (gx < W && gy < H) {
            v = x[((n * C + ic) * H + gy) * W + gx];
        }
        smem[idx] = v;
    }

    __syncthreads();

    const int ox = tile_x + tx;
    const int oy = tile_y + ty;

    if (ox < OW && oy < OH) {
        float acc = b[oc];

        #pragma unroll
        for (int ic = 0; ic < 8; ++ic) {
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    float xv = smem[(ic * tile_h + ty + ky) * tile_w + tx + kx];
                    float wv = w[((oc * C + ic) * K + ky) * K + kx];
                    acc = fmaf(xv, wv, acc);
                }
            }
        }

        acc /= divisor;
        acc = acc > 0.0f ? acc : acc * 0.01f;
        out[((n * O + oc) * OH + oy) * OW + ox] = acc;
    }
}

torch::Tensor conv_div_leaky_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double divisor
) {
    const int N = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int O = weight.size(0);
    const int K = weight.size(2);
    const int OH = H - K + 1;
    const int OW = W - K + 1;

    auto out = torch::empty({N, O, OH, OW}, x.options());

    dim3 block(16, 16);
    dim3 grid((OW + block.x - 1) / block.x, (OH + block.y - 1) / block.y, N * O);
    size_t shared_bytes = C * (block.y + K - 1) * (block.x + K - 1) * sizeof(float);

    conv_div_leaky_kernel<<<grid, block, shared_bytes>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W, O, K, OH, OW,
        static_cast<float>(divisor)
    );

    return out;
}
"""

cpp_sources = r"""
torch::Tensor conv_div_leaky_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double divisor);
"""

conv_div_leaky = load_inline(
    name="conv_div_leaky_inline",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["conv_div_leaky_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.op = conv_div_leaky

    def forward(self, x):
        return self.op.conv_div_leaky_cuda(x, self.conv.weight, self.conv.bias, float(self.divisor))