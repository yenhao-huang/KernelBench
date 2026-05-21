import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void conv_transpose2d_fused_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y,
    int N, int Cin, int H, int W,
    int Cout, int K, int stride,
    int OH, int OW,
    float add_value,
    float multiply_value
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * Cout * OH * OW;
    if (idx >= total) return;

    int ow = idx % OW;
    int t = idx / OW;
    int oh = t % OH;
    t /= OH;
    int oc = t % Cout;
    int n = t / Cout;

    float acc = b[oc];

    for (int ic = 0; ic < Cin; ++ic) {
        for (int kh = 0; kh < K; ++kh) {
            int ih_num = oh - kh;
            if (ih_num < 0 || ih_num % stride != 0) continue;
            int ih = ih_num / stride;
            if (ih < 0 || ih >= H) continue;

            for (int kw = 0; kw < K; ++kw) {
                int iw_num = ow - kw;
                if (iw_num < 0 || iw_num % stride != 0) continue;
                int iw = iw_num / stride;
                if (iw < 0 || iw >= W) continue;

                float xv = x[((n * Cin + ic) * H + ih) * W + iw];
                float wv = w[((ic * Cout + oc) * K + kh) * K + kw];
                acc += xv * wv;
            }
        }
    }

    float v = fminf(acc + add_value, 0.0f);
    float gelu = 0.5f * v * (1.0f + erff(v * 0.7071067811865476f));
    y[idx] = gelu * multiply_value;
}

torch::Tensor conv_transpose2d_fused_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    double add_value,
    double multiply_value
) {
    int N = x.size(0);
    int Cin = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int Cout = weight.size(1);
    int K = weight.size(2);
    int OH = (H - 1) * (int)stride + K;
    int OW = (W - 1) * (int)stride + K;

    auto y = torch::empty({N, Cout, OH, OW}, x.options());

    int total = N * Cout * OH * OW;
    int block = 256;
    int grid = (total + block - 1) / block;

    conv_transpose2d_fused_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        y.data_ptr<float>(),
        N, Cin, H, W, Cout, K, (int)stride, OH, OW,
        (float)add_value,
        (float)multiply_value
    );

    return y;
}
"""

cpp_sources = """
torch::Tensor conv_transpose2d_fused_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    double add_value,
    double multiply_value
);
"""

conv_transpose2d_fused = load_inline(
    name="conv_transpose2d_fused_ext",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["conv_transpose2d_fused_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride
        )
        self.stride = stride
        self.add_value = add_value
        self.multiply_value = multiply_value
        self.op = conv_transpose2d_fused

    def forward(self, x):
        return self.op.conv_transpose2d_fused_cuda(
            x.contiguous(),
            self.conv_transpose.weight.contiguous(),
            self.conv_transpose.bias.contiguous(),
            self.stride,
            self.add_value,
            self.multiply_value,
        )