import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for 3D transposed convolution
conv_transpose3d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int C_out,
    int D, int H, int W,
    int kD, int kH, int kW,
    int strideD, int strideH, int strideW,
    int padD, int padH, int padW,
    int D_out, int H_out, int W_out,
    int C_in_per_group, int C_out_per_group,
    int groups,
    bool use_bias)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C_out * D_out * H_out * W_out;
    if (idx >= total_elements) return;

    // Decompose linear index into n, c_out, d, h, w
    int w = idx % W_out;
    int h = (idx / W_out) % H_out;
    int d = (idx / (W_out * H_out)) % D_out;
    int c_out = (idx / (W_out * H_out * D_out)) % C_out;
    int n = idx / (W_out * H_out * D_out * C_out);

    int group = c_out / C_out_per_group;
    int c_out_in_group = c_out % C_out_per_group;

    float sum = 0.0f;

    // Input channel range for this group
    int c_in_start = group * C_in_per_group;
    int c_in_end = c_in_start + C_in_per_group;

    // Base pointers for this batch and group
    const float* input_n = input + n * C_in * D * H * W;

    for (int c_in = c_in_start; c_in < c_in_end; ++c_in) {
        int c_in_idx = c_in - c_in_start;  // index within group for weight
        const float* input_c = input_n + c_in * D * H * W;

        // Weight pointer for this input channel and output channel within group
        const float* weight_ic = weight + (c_in * C_out_per_group + c_out_in_group) * kD * kH * kW;

        for (int kd = 0; kd < kD; ++kd) {
            int d_in_times_stride = d + padD - kd;
            if (d_in_times_stride % strideD != 0) continue;
            int d_in = d_in_times_stride / strideD;
            if (d_in < 0 || d_in >= D) continue;

            for (int kh = 0; kh < kH; ++kh) {
                int h_in_times_stride = h + padH - kh;
                if (h_in_times_stride % strideH != 0) continue;
                int h_in = h_in_times_stride / strideH;
                if (h_in < 0 || h_in >= H) continue;

                for (int kw = 0; kw < kW; ++kw) {
                    int w_in_times_stride = w + padW - kw;
                    if (w_in_times_stride % strideW != 0) continue;
                    int w_in = w_in_times_stride / strideW;
                    if (w_in < 0 || w_in >= W) continue;

                    float input_val = input_c[(d_in * H + h_in) * W + w_in];
                    float weight_val = weight_ic[(kd * kH + kh) * kW + kw];
                    sum += input_val * weight_val;
                }
            }
        }
    }

    if (use_bias) {
        sum += bias[c_out];
    }

    output[idx] = sum;
}

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::IntArrayRef stride,
    torch::IntArrayRef padding,
    torch::IntArrayRef output_padding,
    int groups)
{
    // Input dimensions
    int N = input.size(0);
    int C_in = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    int C_out = weight.size(1) * groups;  // weight shape: (C_in, C_out/groups, kD, kH, kW)
    int kD = weight.size(2);
    int kH = weight.size(3);
    int kW = weight.size(4);

    int strideD = stride[0];
    int strideH = stride[1];
    int strideW = stride[2];

    int padD = padding[0];
    int padH = padding[1];
    int padW = padding[2];

    int out_padD = output_padding[0];
    int out_padH = output_padding[1];
    int out_padW = output_padding[2];

    // Compute output spatial dimensions
    int D_out = (D - 1) * strideD - 2 * padD + kD + out_padD;
    int H_out = (H - 1) * strideH - 2 * padH + kH + out_padH;
    int W_out = (W - 1) * strideW - 2 * padW + kW + out_padW;

    int C_in_per_group = C_in / groups;
    int C_out_per_group = C_out / groups;

    // Allocate output tensor
    auto output = torch::zeros({N, C_out, D_out, H_out, W_out}, input.options());

    bool use_bias = bias.defined();

    int total_elements = N * C_out * D_out * H_out * W_out;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    conv_transpose3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        use_bias ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, C_out, D, H, W,
        kD, kH, kW,
        strideD, strideH, strideW,
        padD, padH, padW,
        D_out, H_out, W_out,
        C_in_per_group, C_out_per_group,
        groups,
        use_bias
    );

    return output;
}
"""

conv_transpose3d_cpp_source = """
torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::IntArrayRef stride,
    torch::IntArrayRef padding,
    torch::IntArrayRef output_padding,
    int groups);
"""

# Compile the inline CUDA code
conv_transpose3d_op = load_inline(
    name="conv_transpose3d_op",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_cuda_source,
    functions=["conv_transpose3d_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized 3D transposed convolution using a custom CUDA kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0),
                 output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        # Weight and optional bias parameters
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv_transpose3d_op.conv_transpose3d_cuda(
            x, self.weight, self.bias,
            self.stride, self.padding, self.output_padding, self.groups
        )