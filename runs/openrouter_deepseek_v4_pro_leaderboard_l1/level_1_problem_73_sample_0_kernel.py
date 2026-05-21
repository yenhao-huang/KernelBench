import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA source for ConvTranspose3d
conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out, int D_out, int H_out, int W_out,
    int kernel_size, int stride, int padding, int groups,
    int K_D, int K_H, int K_W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C_out * D_out * H_out * W_out;
    if (idx >= total_elements) return;

    // Unravel index: output has shape (N, C_out, D_out, H_out, W_out)
    int w = idx % W_out;
    int h = (idx / W_out) % H_out;
    int d = (idx / (W_out * H_out)) % D_out;
    int c_out = (idx / (W_out * H_out * D_out)) % C_out;
    int n = idx / (W_out * H_out * D_out * C_out);

    int group = c_out / (C_out / groups);
    int c_out_group = c_out % (C_out / groups);

    int C_in_per_group = C_in / groups;

    float value = 0.0f;
    for (int c_in = group * C_in_per_group; c_in < (group + 1) * C_in_per_group; ++c_in) {
        int c_in_group = c_in % C_in_per_group;
        // weight layout: (C_in, C_out/groups, K_D, K_H, K_W)
        int weight_idx_base = ((c_in * (C_out / groups) + c_out_group) * K_D * K_H * K_W);

        for (int kd = 0; kd < K_D; ++kd) {
            int in_d = d * stride + kd - padding;
            if (in_d < 0 || in_d >= D_in) continue;
            for (int kh = 0; kh < K_H; ++kh) {
                int in_h = h * stride + kh - padding;
                if (in_h < 0 || in_h >= H_in) continue;
                for (int kw = 0; kw < K_W; ++kw) {
                    int in_w = w * stride + kw - padding;
                    if (in_w < 0 || in_w >= W_in) continue;

                    int input_idx = ((n * C_in + c_in) * D_in + in_d) * H_in + in_h;
                    input_idx = input_idx * W_in + in_w;
                    int weight_idx = weight_idx_base + (kd * K_H + kh) * K_W + kw;
                    value += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    if (bias != nullptr) {
        value += bias[c_out];
    }
    output[idx] = value;
}

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups
) {
    // Input shape: (N, C_in, D_in, H_in, W_in)
    const auto in_shape = input.sizes();
    int N = in_shape[0];
    int C_in = in_shape[1];
    int D_in = in_shape[2];
    int H_in = in_shape[3];
    int W_in = in_shape[4];

    // Weight shape: (C_in, C_out/groups, K_D, K_H, K_W)
    const auto w_shape = weight.sizes();
    int K_D = w_shape[2];
    int K_H = w_shape[3];
    int K_W = w_shape[4];
    int C_out_groups = w_shape[1]; // out channels per group
    int C_out = C_out_groups * groups;

    // Compute output spatial sizes
    int D_out = (D_in - 1) * stride - 2 * padding + K_D + output_padding;
    int H_out = (H_in - 1) * stride - 2 * padding + K_H + output_padding;
    int W_out = (W_in - 1) * stride - 2 * padding + K_W + output_padding;

    auto output = torch::zeros({N, C_out, D_out, H_out, W_out}, input.options());

    const int block_size = 256;
    int total_elements = N * C_out * D_out * H_out * W_out;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;

    conv_transpose3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, C_in, D_in, H_in, W_in,
        C_out, D_out, H_out, W_out,
        kernel_size, stride, padding, groups,
        K_D, K_H, K_W
    );

    return output;
}
"""

conv_transpose3d_cpp_source = (
    "torch::Tensor conv_transpose3d_cuda(torch::Tensor input, torch::Tensor weight, torch::optional<torch::Tensor> bias, int kernel_size, int stride, int padding, int output_padding, int groups);"
)

# Compile the inline CUDA code
conv_transpose3d_op = load_inline(
    name="conv_transpose3d_cuda_op",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Custom ConvTranspose3d implemented with inline CUDA.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias_flag: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias_flag = bias_flag

        # Weight parameter shape: (in_channels, out_channels/group, kernel_size, kernel_size, kernel_size)
        weight_shape = (in_channels, out_channels // groups, kernel_size, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.empty(*weight_shape))
        if bias_flag:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weight using Kaiming uniform
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv_transpose3d_op.conv_transpose3d_cuda(
            x, self.weight, self.bias,
            self.kernel_size, self.stride, self.padding,
            self.output_padding, self.groups
        )