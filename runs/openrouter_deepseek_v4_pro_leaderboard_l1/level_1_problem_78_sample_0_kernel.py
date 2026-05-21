import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for ConvTranspose2d
conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_kernel(
    const float* input, const float* weight, const float* bias, float* output,
    int N, int C_in, int C_out, int H_in, int W_in, int H_out, int W_out,
    int kernel_h, int kernel_w, int stride_h, int stride_w, int padding_h, int padding_w,
    bool has_bias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C_out * H_out * W_out;
    if (idx >= total_elements) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c_out = (idx / (W_out * H_out)) % C_out;
    int n = idx / (W_out * H_out * C_out);

    float value = 0.0f;

    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h_in = h_out + padding_h - kh;
                int w_in = w_out + padding_w - kw;

                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    int input_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                    int weight_idx = ((c_in * C_out + c_out) * kernel_h + kh) * kernel_w + kw;
                    value += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    if (has_bias) {
        value += bias[c_out];
    }

    output[idx] = value;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    torch::IntArrayRef stride,
    torch::IntArrayRef padding
) {
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);

    const int C_out = weight.size(1);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);

    int stride_h = stride[0];
    int stride_w = stride[1];
    int padding_h = padding[0];
    int padding_w = padding[1];

    int H_out = (H_in - 1) * stride_h - 2 * padding_h + kernel_h;
    int W_out = (W_in - 1) * stride_w - 2 * padding_w + kernel_w;

    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    const int total_elements = N * C_out * H_out * W_out;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    bool has_bias = bias.has_value();
    const float* bias_ptr = has_bias ? bias.value().data_ptr<float>() : nullptr;

    conv_transpose2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr, output.data_ptr<float>(),
        N, C_in, C_out, H_in, W_in, H_out, W_out,
        kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w,
        has_bias
    );

    return output;
}
"""

conv_transpose2d_cpp_source = (
    "torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, torch::optional<torch::Tensor> bias, torch::IntArrayRef stride, torch::IntArrayRef padding);"
)

conv_transpose2d_cuda = load_inline(
    name="conv_transpose2d_cuda",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.has_bias = bias

        # Weight shape: (in_channels, out_channels, kh, kw) matching PyTorch's ConvTranspose2d
        weight = torch.empty(in_channels, out_channels, *kernel_size)
        self.weight = nn.Parameter(weight)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv_transpose2d_cuda.conv_transpose2d_cuda(x, self.weight, self.bias, self.stride, self.padding)