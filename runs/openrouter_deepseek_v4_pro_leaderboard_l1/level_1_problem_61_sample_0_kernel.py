```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for transposed 3D convolution
conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int depth, int height, int width,
    int out_depth, int out_height, int out_width,
    int kernel_d, int kernel_h, int kernel_w,
    int stride, int padding, int output_padding,
    int groups,
    bool has_bias) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    if (idx >= total_elements) return;

    // Decode output index
    int n = idx / (out_channels * out_depth * out_height * out_width);
    int rem = idx % (out_channels * out_depth * out_height * out_width);
    int oc = rem / (out_depth * out_height * out_width);
    rem = rem % (out_depth * out_height * out_width);
    int d = rem / (out_height * out_width);
    rem = rem % (out_height * out_width);
    int h = rem / out_width;
    int w = rem % out_width;

    int group_out_channels = out_channels / groups;
    int group_in_channels = in_channels / groups;
    int g = oc / group_out_channels;
    int ic_start = g * group_in_channels;
    int ic_end = ic_start + group_in_channels;
    int oc_group = oc % group_out_channels;

    float val = 0.0f;

    for (int ic = ic_start; ic < ic_end; ++ic) {
        for (int kd = 0; kd < kernel_d; ++kd) {
            int d_i = d + padding - kd;
            if (d_i % stride != 0) continue;
            d_i /= stride;
            if (d_i < 0 || d_i >= depth) continue;

            for (int kh = 0; kh < kernel_h; ++kh) {
                int h_i = h + padding - kh;
                if (h_i % stride != 0) continue;
                h_i /= stride;
                if (h_i < 0 || h_i >= height) continue;

                for (int kw = 0; kw < kernel_w; ++kw) {
                    int w_i = w + padding - kw;
                    if (w_i % stride != 0) continue;
                    w_i /= stride;
                    if (w_i < 0 || w_i >= width) continue;

                    // weight layout: (in_channels, out_channels/groups, D, H, W)
                    int weight_idx = ((ic * group_out_channels + oc_group) * kernel_d * kernel_h * kernel_w) +
                                     ((kd * kernel_h + kh) * kernel_w + kw);
                    float w_val = weight[weight_idx];

                    int input_idx = (((n * in_channels + ic) * depth + d_i) * height + h_i) * width + w_i;
                    float in_val = input[input_idx];

                    val += in_val * w_val;
                }
            }
        }
    }

    if (has_bias) {
        val += bias[oc];
    }

    output[idx] = val;
}

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    bool has_bias,
    int stride,
    int padding,
    int output_padding,
    int groups) {

    input = input.contiguous();
    weight = weight.contiguous();

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);

    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);
    int out_channels = weight.size(1) * groups;

    int out_depth = (depth - 1) * stride - 2 * padding + kernel_d + output_padding;
    int out_height = (height - 1) * stride - 2 * padding + kernel_h + output_padding;
    int out_width = (width - 1) * stride - 2 * padding + kernel_w + output_padding;

    auto out = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width}, input.options());

    int total_threads = batch_size * out_channels * out_depth * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total_threads + block_size - 1) / block_size;

    conv_transpose3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        has_bias ? bias.data_ptr<float>() : nullptr,
        out.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        depth, height, width,
        out_depth, out_height, out_width,
        kernel_d, kernel_h, kernel_w,
        stride, padding, output_padding,
        groups,
        has_bias
    );

    return out;
}
"""

conv_transpose3d_cpp_source = """
torch::Tensor conv_transpose3d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, bool has_bias, int stride, int padding, int output_padding, int groups);
"""

# Compile the inline CUDA code
conv_transpose3d_module = load_inline(
    name="conv_transpose3d_module",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_cuda"],
    verbose=True,
    extra_cflags=[],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.has_b