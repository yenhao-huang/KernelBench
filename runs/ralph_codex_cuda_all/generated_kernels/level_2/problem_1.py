import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cpp_sources = """
torch::Tensor conv_relu_add_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor conv_b, torch::Tensor add_b);
"""

cuda_sources = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv3x3_relu_add_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ conv_b,
    const float* __restrict__ add_b,
    float* __restrict__ out,
    int N, int C, int H, int W, int O, int OH, int OW
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * O * OH * OW;
    if (idx >= total) return;

    int ow = idx % OW;
    int t = idx / OW;
    int oh = t % OH;
    t /= OH;
    int oc = t % O;
    int n = t / O;

    float acc = conv_b[oc];

    const int x_n_base = n * C * H * W;
    const int w_o_base = oc * C * 9;

    #pragma unroll
    for (int c = 0; c < 64; ++c) {
        int x_base = x_n_base + c * H * W + oh * W + ow;
        int w_base = w_o_base + c * 9;

        acc += x[x_base] * w[w_base];
        acc += x[x_base + 1] * w[w_base + 1];
        acc += x[x_base + 2] * w[w_base + 2];

        acc += x[x_base + W] * w[w_base + 3];
        acc += x[x_base + W + 1] * w[w_base + 4];
        acc += x[x_base + W + 2] * w[w_base + 5];

        acc += x[x_base + 2 * W] * w[w_base + 6];
        acc += x[x_base + 2 * W + 1] * w[w_base + 7];
        acc += x[x_base + 2 * W + 2] * w[w_base + 8];
    }

    acc = acc > 0.0f ? acc : 0.0f;
    out[idx] = acc + add_b[oc];
}

__global__ void convk_relu_add_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ conv_b,
    const float* __restrict__ add_b,
    float* __restrict__ out,
    int N, int C, int H, int W, int O, int K, int OH, int OW
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * O * OH * OW;
    if (idx >= total) return;

    int ow = idx % OW;
    int t = idx / OW;
    int oh = t % OH;
    t /= OH;
    int oc = t % O;
    int n = t / O;

    float acc = conv_b[oc];

    for (int c = 0; c < C; ++c) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int xi = ((n * C + c) * H + oh + kh) * W + ow + kw;
                int wi = ((oc * C + c) * K + kh) * K + kw;
                acc += x[xi] * w[wi];
            }
        }
    }

    acc = acc > 0.0f ? acc : 0.0f;
    out[idx] = acc + add_b[oc];
}

torch::Tensor conv_relu_add_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor conv_b, torch::Tensor add_b) {
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int O = w.size(0);
    int K = w.size(2);
    int OH = H - K + 1;
    int OW = W - K + 1;

    auto out = torch::empty({N, O, OH, OW}, x.options());

    int total = N * O * OH * OW;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    if (K == 3 && C == 64) {
        conv3x3_relu_add_kernel<<<blocks, threads>>>(
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            conv_b.data_ptr<float>(),
            add_b.data_ptr<float>(),
            out.data_ptr<float>(),
            N, C, H, W, O, OH, OW
        );
    } else {
        convk_relu_add_kernel<<<blocks, threads>>>(
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            conv_b.data_ptr<float>(),
            add_b.data_ptr<float>(),
            out.data_ptr<float>(),
            N, C, H, W, O, K, OH, OW
        );
    }

    return out;
}
"""

conv_relu_add_ext = load_inline(
    name="conv_relu_add_ext",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["conv_relu_add_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.op = conv_relu_add_ext

    def forward(self, x):
        return self.op.conv_relu_add_cuda(x, self.conv.weight, self.conv.bias, self.bias)