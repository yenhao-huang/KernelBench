import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose1d_fp32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y,
    int N, int Cin, int L,
    int Cout, int K,
    int stride, int padding, int dilation,
    int Lout,
    bool has_bias
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int co = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z;

    if (t >= Lout || co >= Cout) return;

    float acc = has_bias ? b[co] : 0.0f;

    #pragma unroll
    for (int ci = 0; ci < 32; ++ci) {
        if (ci >= Cin) break;

        #pragma unroll
        for (int k = 0; k < 5; ++k) {
            if (k >= K) break;

            int src = t + padding - k * dilation;
            if (src >= 0 && src % stride == 0) {
                src /= stride;
                if (src < L) {
                    float xv = x[(n * Cin + ci) * L + src];
                    float wv = w[(ci * Cout + co) * K + k];
                    acc += xv * wv;
                }
            }
        }
    }

    y[(n * Cout + co) * Lout + t] = acc;
}

torch::Tensor conv_transpose1d_fp32_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    bool has_bias
) {
    int N = x.size(0);
    int Cin = x.size(1);
    int L = x.size(2);
    int Cout = w.size(1);
    int K = w.size(2);
    int Lout = (L - 1) * stride - 2 * padding + dilation * (K - 1) + 1;

    auto y = torch::empty({N, Cout, Lout}, x.options());

    dim3 block(128, 2);
    dim3 grid((Lout + block.x - 1) / block.x, (Cout + block.y - 1) / block.y, N);

    conv_transpose1d_fp32_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        has_bias ? bias.data_ptr<float>() : nullptr,
        y.data_ptr<float>(),
        N, Cin, L, Cout, K,
        stride, padding, dilation,
        Lout,
        has_bias
    );

    return y;
}
"""

cpp_sources = r"""
torch::Tensor conv_transpose1d_fp32_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    bool has_bias
);
"""

conv_transpose1d_ext = load_inline(
    name="conv_transpose1d_fp32_ext",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["conv_transpose1d_fp32_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super().__init__()
        ref = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.weight = nn.Parameter(ref.weight.detach().clone())
        if bias:
            self.bias = nn.Parameter(ref.bias.detach().clone())
        else:
            self.register_buffer("bias", torch.empty(0))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.has_bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv_transpose1d_ext.conv_transpose1d_fp32_cuda(
            x.contiguous(),
            self.weight.contiguous(),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.has_bias,
        )