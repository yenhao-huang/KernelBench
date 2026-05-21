import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source code for transposed 2D convolution (scatter approach with atomicAdd)
conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_scatter_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int groups
) {
    int total_input_elements = batch_size * in_channels * in_h * in_w;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_total = gridDim.x * blockDim.x;

    int in_channels_per_group = in_channels / groups;
    int out_channels_per_group = out_channels / groups;

    for (int i = idx; i < total_input_elements; i += stride_total) {
        int w = i % in_w;
        int h = (i / in_w) % in_h;
        int ic = (i / (in_w * in_h)) % in_channels;
        int n = i / (in_w * in_h * in_channels);

        float input_val = input[i];
        int group = ic / in_channels_per_group;
        int oc_start = group * out_channels_per_group;
        int oc_end = oc_start + out_channels_per_group;

        for (int oc = oc_start; oc < oc_end; ++oc) {
            int oc_local = oc - oc_start;
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int oh = h * stride_h + kh - padding_h;
                    int ow = w * stride_w + kw - padding_w;
                    if (oh >= 0 && oh < out_h && ow >= 0 && ow < out_w) {
                        int weight_idx = ((ic * out_channels_per_group + oc_local) * kernel_h + kh) * kernel_w + kw;
                        float w_val = weight[weight_idx];
                        int out_idx = ((n * out_channels + oc) * out_h + oh) * out_w + ow;
                        atomicAdd(&output[out_idx], input_val * w_val);
                    }
                }
            }
        }
    }
}

__global__ void add_bias_kernel(float* output, const float* bias, int out_channels, int out_h, int out_w, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_channels * out_h * out_w;
    int stride_total = gridDim.x * blockDim.x;
    for (int i = idx; i < total; i += stride_total) {
        int oc = (i / (out_h * out_w)) % out_channels;
        output[i] += bias[oc];
    }
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int output_padding_h, int output_padding_w,
    int groups) {

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);
    int out_channels = weight.size(1) * groups; // weight shape: (in_channels, out_channels/groups, kH, kW)
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    int out_h = (in_h - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h;
    int out_w = (in_w - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w;

    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, input.options());

    const int block_size = 256;
    const int num_blocks = 1024; // enough parallelism

    conv_transpose2d_scatter_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_h, in_w, out_h, out_w,
        kernel_h, kernel_w,
        stride_h, stride_w,
        padding_h, padding_w,
        groups
    );

    if (bias.numel() > 0) {
        add_bias_kernel<<<num_blocks, block_size>>>(
            output.data_ptr<float>(),
            bias.data_ptr<float>(),
            out_channels, out_h, out_w, batch_size
        );
    }

    return output;
}
"""

conv_transpose2d_cpp_source = """
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int output_padding_h, int output_padding_w,
    int groups);
"""

# Compile the inline CUDA code
conv_transpose2d = load_inline(
    name="conv_transpose2d",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding)
        self.groups = groups
        self.use_bias = bias

        # Weight shape: (in_channels, out_channels // groups, kernel_h, kernel_w)
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias if self.bias is not None else torch.empty(0, device=x.device)
        return conv_transpose2d.conv_transpose2d_cuda(
            x, self.weight, bias,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.output_padding[0], self.output_padding[1],
            self.groups
        )