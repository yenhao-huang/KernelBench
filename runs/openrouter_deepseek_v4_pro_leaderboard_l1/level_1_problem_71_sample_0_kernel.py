import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA source for the custom transposed 2D convolution kernel
conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>

__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* output,
    const float* __restrict__ bias,
    int N, int C_in, int C_out, int H_in, int W_in, int H_out, int W_out,
    int kernel_size, int stride, int padding, int output_padding, int groups,
    int C_out_per_group, int C_in_per_group)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H_out * W_out;
    int stride_total = gridDim.x * blockDim.x;

    for (int i = idx; i < total; i += stride_total) {
        int n = i / (C_out * H_out * W_out);
        int rem = i % (C_out * H_out * W_out);
        int c_out = rem / (H_out * W_out);
        int rem2 = rem % (H_out * W_out);
        int h_out = rem2 / W_out;
        int w_out = rem2 % W_out;

        int group = c_out / C_out_per_group;
        int c_out_in_group = c_out % C_out_per_group;

        float acc = 0.0f;

        // Gather operation: accumulate contributions from all relevant input positions and filter taps
        for (int fh = 0; fh < kernel_size; ++fh) {
            int h_base = h_out + padding - fh;
            if (h_base % stride == 0) {
                int h_in = h_base / stride;
                if (h_in >= 0 && h_in < H_in) {
                    for (int fw = 0; fw < kernel_size; ++fw) {
                        int w_base = w_out + padding - fw;
                        if (w_base % stride == 0) {
                            int w_in = w_base / stride;
                            if (w_in >= 0 && w_in < W_in) {
                                for (int c_in = group * C_in_per_group; c_in < (group + 1) * C_in_per_group; ++c_in) {
                                    float in_val = input[((n * C_in + c_in) * H_in + h_in) * W_in + w_in];
                                    float w_val = weight[((c_in * C_out_per_group + c_out_in_group) * kernel_size + fh) * kernel_size + fw];
                                    acc += in_val * w_val;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Optional bias addition
        if (bias != nullptr) {
            acc += bias[c_out];
        }

        output[i] = acc;
    }
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int stride, int padding, int output_padding, int groups)
{
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);
    const int kernel_size = weight.size(2);  // square kernel
    const int C_out = weight.size(1) * groups;  // weight shape: (C_in, C_out_per_group, k, k)

    // Output spatial dimensions according to ConvTranspose2d formula
    int H_out = (H_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    int W_out = (W_in - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    const int C_out_per_group = C_out / groups;
    const int C_in_per_group = C_in / groups;

    const int block_size = 256;
    const int total_elements = N * C_out * H_out * W_out;
    const int num_blocks = std::min(1024, (total_elements + block_size - 1) / block_size);

    // Launch kernel, passing bias pointer if it exists
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    conv_transpose2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
        bias_ptr,
        N, C_in, C_out, H_in, W_in, H_out, W_out, kernel_size, stride, padding, output_padding, groups,
        C_out_per_group, C_in_per_group);

    return output;
}
"""

conv_transpose2d_cpp_source = """
#include <torch/extension.h>
#include <c10/util/Optional.h>
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int stride, int padding, int output_padding, int groups);
"""

# Compile the inline CUDA code
conv_transpose2d_op = load_inline(
    name="conv_transpose2d_op",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    """
    Optimized model that replaces nn.ConvTranspose2d with a custom CUDA kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias_flag = bias

        # Weight shape matches the internal shape of nn.ConvTranspose2d: (in_channels, out_channels/groups, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        # Mimic the default initialization of nn.ConvTranspose2d
        nn.init.kaiming_uniform_(self.weight, a=5.0**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Call the custom CUDA kernel
        return conv_transpose2d_op.conv_transpose2d_cuda(
            x,
            self.weight,
            self.bias if self.bias_flag else None,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups
        )