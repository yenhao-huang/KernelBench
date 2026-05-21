import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for 3D transposed convolution
conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int C_out,
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out,
    int K, int stride, int padding, int dilation) {

    int batch = blockIdx.z;
    int out_channel = blockIdx.y;
    int spatial_start = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch >= N || out_channel >= C_out) return;

    int total_spatial = D_out * H_out * W_out;
    if (spatial_start >= total_spatial) return;

    int od = spatial_start / (H_out * W_out);
    int rem = spatial_start % (H_out * W_out);
    int oh = rem / W_out;
    int ow = rem % W_out;

    float accum = 0.0f;

    for (int ic = 0; ic < C_in; ++ic) {
        for (int kd = 0; kd < K; ++kd) {
            int d_in = od * stride - padding + kd * dilation;
            if (d_in < 0 || d_in >= D_in) continue;
            for (int kh = 0; kh < K; ++kh) {
                int h_in = oh * stride - padding + kh * dilation;
                if (h_in < 0 || h_in >= H_in) continue;
                for (int kw = 0; kw < K; ++kw) {
                    int w_in = ow * stride - padding + kw * dilation;
                    if (w_in < 0 || w_in >= W_in) continue;

                    int input_idx = (((batch * C_in + ic) * D_in + d_in) * H_in + h_in) * W_in + w_in;
                    int weight_idx = (((out_channel * C_in + ic) * K + kd) * K + kh) * K + kw;

                    accum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    if (bias != nullptr) {
        accum += bias[out_channel];
    }

    int output_idx = (((batch * C_out + out_channel) * D_out + od) * H_out + oh) * W_out + ow;
    output[output_idx] = accum;
}

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride, int padding, int dilation, int output_padding) {

    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");

    const int N = input.size(0);
    const int C_in = input.size(1);
    const int D_in = input.size(2);
    const int H_in = input.size(3);
    const int W_in = input.size(4);
    const int C_out = weight.size(0);
    const int K = weight.size(2); // kernel size (square)

    // Compute output spatial dimensions
    const int D_out = (D_in - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1;
    const int H_out = (H_in - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1;
    const int W_out = (W_in - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1;

    auto output = torch::empty({N, C_out, D_out, H_out, W_out}, input.options());

    const int total_spatial = D_out * H_out * W_out;
    const int block_size = 256;
    const int grid_x = (total_spatial + block_size - 1) / block_size;

    dim3 grid(grid_x, C_out, N);
    dim3 block(block_size);

    const float* bias_ptr = bias.has_value() ? bias->data_ptr<float>() : nullptr;

    conv_transpose3d_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, C_in, C_out,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        K, stride, padding, dilation);

    return output;
}
"""

conv_transpose3d_cpp_source = """
torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride, int padding, int dilation, int output_padding);
"""

# Compile the inline CUDA code
custom_conv_transpose3d = load_inline(
    name="custom_conv_transpose3d",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = 0  # default not provided, assume 0

        # Initialize weight
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure tensors are contiguous
        x = x.contiguous()
        weight = self.weight.contiguous()
        bias = self.bias.contiguous() if self.bias is not None else torch.empty(0)

        return custom_conv_transpose3d.conv_transpose3d_cuda(
            x, weight, bias,
            self.stride, self.padding, self.dilation, self.output_padding
        )