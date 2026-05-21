import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for transposed 3D convolution
conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_depth,
    const int in_width,
    const int in_height,
    const int kernel_depth,
    const int kernel_width,
    const int kernel_height,
    const int stride_depth,
    const int stride_width,
    const int stride_height,
    const int padding_depth,
    const int padding_width,
    const int padding_height,
    const int output_padding_depth,
    const int output_padding_width,
    const int output_padding_height,
    const int out_depth,
    const int out_width,
    const int out_height,
    const int groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_depth * out_width * out_height;
    if (idx >= total_elements) return;

    // Decompose linear index into batch, channel, and spatial indices
    int n = idx / (out_channels * out_depth * out_width * out_height);
    int rem = idx % (out_channels * out_depth * out_width * out_height);
    int oc = rem / (out_depth * out_width * out_height);
    rem = rem % (out_depth * out_width * out_height);
    int od = rem / (out_width * out_height);
    rem = rem % (out_width * out_height);
    int ow = rem / out_height;
    int oh = rem % out_height;

    int group_size_in = in_channels / groups;
    int group_size_out = out_channels / groups;
    int group_id = oc / group_size_out;
    int oc_in_group = oc % group_size_out;

    float sum = 0.0f;

    // Iterate over input channels in the group
    for (int ic = group_id * group_size_in; ic < (group_id + 1) * group_size_in; ++ic) {
        // Iterate over kernel spatial dimensions
        for (int kd = 0; kd < kernel_depth; ++kd) {
            int id = od + padding_depth - kd;
            if (id < 0 || id >= in_depth * stride_depth || id % stride_depth != 0) continue;
            id /= stride_depth;
            if (id >= in_depth) continue;

            for (int kw = 0; kw < kernel_width; ++kw) {
                int iw = ow + padding_width - kw;
                if (iw < 0 || iw >= in_width * stride_width || iw % stride_width != 0) continue;
                iw /= stride_width;
                if (iw >= in_width) continue;

                for (int kh = 0; kh < kernel_height; ++kh) {
                    int ih = oh + padding_height - kh;
                    if (ih < 0 || ih >= in_height * stride_height || ih % stride_height != 0) continue;
                    ih /= stride_height;
                    if (ih >= in_height) continue;

                    int input_idx = ((n * in_channels + ic) * in_depth + id) * in_width + iw;
                    input_idx = input_idx * in_height + ih;
                    int weight_idx = ((oc * in_channels / groups + (ic - group_id * group_size_in)) * kernel_depth + kd) * kernel_width + kw;
                    weight_idx = weight_idx * kernel_height + kh;

                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    output[idx] = sum;
}

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::IntArrayRef stride,
    torch::IntArrayRef padding,
    torch::IntArrayRef output_padding,
    int groups
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_depth = input.size(2);
    const int in_width = input.size(3);
    const int in_height = input.size(4);
    const int out_channels = weight.size(0);
    const int kernel_depth = weight.size(2);
    const int kernel_width = weight.size(3);
    const int kernel_height = weight.size(4);

    int stride_depth = stride[0];
    int stride_width = stride[1];
    int stride_height = stride[2];
    int padding_depth = padding[0];
    int padding_width = padding[1];
    int padding_height = padding[2];
    int output_padding_depth = output_padding[0];
    int output_padding_width = output_padding[1];
    int output_padding_height = output_padding[2];

    int out_depth = (in_depth - 1) * stride_depth - 2 * padding_depth + kernel_depth + output_padding_depth;
    int out_width = (in_width - 1) * stride_width - 2 * padding_width + kernel_width + output_padding_width;
    int out_height = (in_height - 1) * stride_height - 2 * padding_height + kernel_height + output_padding_height;

    auto output = torch::zeros({batch_size, out_channels, out_depth, out_width, out_height}, input.options());

    int total_elements = batch_size * out_channels * out_depth * out_width * out_height;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    conv_transpose3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_depth,
        in_width,
        in_height,
        kernel_depth,
        kernel_width,
        kernel_height,
        stride_depth,
        stride_width,
        stride_height,
        padding_depth,
        padding_width,
        padding_height,
        output_padding_depth,
        output_padding_width,
        output_padding_height,
        out_depth,
        out_width,
        out_height,
        groups
    );

    return output;
}
"""

conv_transpose3d_cpp_source = (
    "torch::Tensor conv_transpose3d_cuda("
    "   torch::Tensor input,"
    "   torch::Tensor weight,"
    "   torch::IntArrayRef stride,"
    "   torch::IntArrayRef padding,"
    "   torch::IntArrayRef output_padding,"
    "   int groups"
    ");"
)

# Compile the inline CUDA code
conv_transpose3d = load_inline(
    name="conv_transpose3d",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias

        # Initialize weight and optional bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.conv_transpose3d = conv_transpose3d

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_transpose3d.conv_transpose3d_cuda(
            x,
            self.weight,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups
        )
        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1, 1)
        return out