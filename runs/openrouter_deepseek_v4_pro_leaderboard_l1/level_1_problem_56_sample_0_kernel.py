import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source code for custom 2D convolution (forward only)
conv2d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>

__global__ void conv2d_direct_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int K_out, int K_in, int kH, int kW,
    int outH, int outW,
    int padH, int padW,
    int strideH, int strideW,
    int dilationH, int dilationW,
    int groups, bool has_bias)
{
    int total_outs = N * K_out * outH * outW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_outs) return;

    int n = idx / (K_out * outH * outW);
    int rem = idx % (K_out * outH * outW);
    int k = rem / (outH * outW);
    int rem2 = rem % (outH * outW);
    int oh = rem2 / outW;
    int ow = rem2 % outW;

    int inC_per_group = C / groups;
    int outC_per_group = K_out / groups;

    int g = k / outC_per_group;
    int ic_start = g * inC_per_group;
    int ic_end = ic_start + inC_per_group;

    float sum = 0.0f;
    for (int ic = ic_start; ic < ic_end; ++ic) {
        for (int kh = 0; kh < kH; ++kh) {
            for (int kw = 0; kw < kW; ++kw) {
                int ih = oh * strideH - padH + kh * dilationH;
                int iw = ow * strideW - padW + kw * dilationW;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    float val = input[n * C * H * W + ic * H * W + ih * W + iw];
                    float w = weight[k * K_in * kH * kW + (ic - ic_start) * kH * kW + kh * kW + kw];
                    sum += val * w;
                }
            }
        }
    }
    if (has_bias) sum += bias[k];
    output[idx] = sum;
}

torch::Tensor conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int strideH, int strideW,
    int padH, int padW,
    int dilationH, int dilationW,
    int groups) {

    // Ensure input is contiguous float tensor on CUDA
    input = input.contiguous();
    weight = weight.contiguous();

    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N, C, H, W)");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D (outC, inC/groups, kH, kW)");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(weight.device().is_cuda(), "Weight must be on CUDA");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int K_out = weight.size(0);
    int K_in = weight.size(1); // in_channels_per_group
    int kH = weight.size(2);
    int kW = weight.size(3);

    // Compute output spatial dimensions
    int outH = (H + 2 * padH - dilationH * (kH - 1) - 1) / strideH + 1;
    int outW = (W + 2 * padW - dilationW * (kW - 1) - 1) / strideW + 1;

    auto output = torch::zeros({N, K_out, outH, outW}, input.options());

    const float* bias_ptr = nullptr;
    bool has_bias = bias.has_value();
    if (has_bias) {
        bias_ptr = bias->contiguous().data_ptr<float>();
    }

    int total_outs = N * K_out * outH * outW;
    const int block_size = 256;
    const int num_blocks = (total_outs + block_size - 1) / block_size;

    conv2d_direct_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, C, H, W,
        K_out, K_in, kH, kW,
        outH, outW,
        padH, padW,
        strideH, strideW,
        dilationH, dilationW,
        groups, has_bias);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}
"""

conv2d_cpp_source = """
torch::Tensor conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int strideH, int strideW,
    int padH, int padW,
    int dilationH, int dilationW,
    int groups);
"""

# Compile the inline CUDA code for custom convolution
conv2d_custom = load_inline(
    name="conv2d_custom",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_cuda_source,
    functions=["conv2d_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized 2D convolution using a custom CUDA kernel (forward only).
    Replaces nn.Conv2d with a direct convolution implementation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.has_bias = bias

        # Weight shape: (out_channels, in_channels//groups, kernel_height, kernel_width)
        weight_shape = (out_channels, in_channels // groups, kernel_size[0], kernel_size[1])
        self.weight = nn.Parameter(torch.empty(*weight_shape), requires_grad=False)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels), requires_grad=False)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        # Store the custom module
        self.conv2d_cuda = conv2d_custom

    def reset_parameters(self):
        # Initialize weight and optional bias as nn.Conv2d would
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.has_bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return self.conv2d_cuda.conv2d_cuda(
            x,
            self.weight,
            self.bias if self.has_bias else None,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1],
            self.groups
        )