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
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int depth_in, int height_in, int width_in,
    int depth_out, int height_out, int width_out,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups,
    bool use_bias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * depth_out * height_out * width_out;
    if (idx >= total_elements) return;

    // Decompose linear index
    int w_out = idx % width_out;
    int h_out = (idx / width_out) % height_out;
    int d_out = (idx / (width_out * height_out)) % depth_out;
    int oc = (idx / (width_out * height_out * depth_out)) % out_channels;
    int b = idx / (width_out * height_out * depth_out * out_channels);

    int channels_per_group = in_channels / groups;
    int out_channels_per_group = out_channels / groups;
    int group = oc / out_channels_per_group;
    int oc_in_group = oc % out_channels_per_group;

    float sum = 0.0f;

    for (int ic = group * channels_per_group; ic < (group + 1) * channels_per_group; ++ic) {
        for (int kd = 0; kd < kernel_d; ++kd) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int in_d = d_out + padding_d - kd * dilation_d;
                    int in_h = h_out + padding_h - kh * dilation_h;
                    int in_w = w_out + padding_w - kw * dilation_w;

                    if (in_d % stride_d == 0 && in_h % stride_h == 0 && in_w % stride_w == 0) {
                        in_d /= stride_d;
                        in_h /= stride_h;
                        in_w /= stride_w;

                        if (in_d >= 0 && in_d < depth_in && in_h >= 0 && in_h < height_in && in_w >= 0 && in_w < width_in) {
                            float input_val = input[((b * in_channels + ic) * depth_in + in_d) * height_in * width_in + in_h * width_in + in_w];
                            float weight_val = weight[((oc * in_channels / groups + (ic - group * channels_per_group)) * kernel_d + kd) * kernel_h * kernel_w + kh * kernel_w + kw];
                            sum += input_val * weight_val;
                        }
                    }
                }
            }
        }
    }

    if (use_bias) {
        sum += bias[oc];
    }

    output[idx] = sum;
}

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int depth_in = input.size(2);
    int height_in = input.size(3);
    int width_in = input.size(4);

    int out_channels = weight.size(0);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);

    int depth_out = (depth_in - 1) * stride_d - 2 * padding_d + dilation_d * (kernel_d - 1) + output_padding_d + 1;
    int height_out = (height_in - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_h - 1) + output_padding_h + 1;
    int width_out = (width_in - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_w - 1) + output_padding_w + 1;

    auto output = torch::zeros({batch_size, out_channels, depth_out, height_out, width_out}, input.options());

    int total_elements = batch_size * out_channels * depth_out * height_out * width_out;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    bool use_bias = bias.numel() > 0;

    conv_transpose3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        use_bias ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        depth_in, height_in, width_in,
        depth_out, height_out, width_out,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w,
        dilation_d, dilation_h, dilation_w,
        groups,
        use_bias
    );

    return output;
}
"""

conv_transpose3d_cpp_source = (
    "torch::Tensor conv_transpose3d_cuda("
    "torch::Tensor input,"
    "torch::Tensor weight,"
    "torch::Tensor bias,"
    "int stride_d, int stride_h, int stride_w,"
    "int padding_d, int padding_h, int padding_w,"
    "int output_padding_d, int output_padding_h, int output_padding_w,"
    "int dilation_d, int dilation_h, int dilation_w,"
    "int groups"
    ");"
)

# Compile the inline CUDA code
conv_transpose3d_op = load_inline(
    name="conv_transpose3d_op",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding, output_padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
        self.groups = groups
        self.bias_flag = bias

        # Initialize weight and bias as parameters
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.conv_transpose3d_op = conv_transpose3d_op

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_transpose3d_op.conv_transpose3d_cuda(
            x,
            self.weight,
            self.bias if self.bias_flag else torch.empty(0, device=x.device),
            self.stride[0], self.stride[1], self.stride[2],
            self.padding[0], self.padding[1], self.padding[2],
            self.output_padding[0], self.output_padding[1], self.output_padding[2],
            self.dilation[0], self.dilation[1], self.dilation[2],
            self.groups
        )