import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for fused depthwise-separable convolution
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_depthwise_separable_kernel(
    const float* __restrict__ input,
    const float* __restrict__ depthwise_weight,
    const float* __restrict__ pointwise_weight,
    const float* __restrict__ depthwise_bias,
    const float* __restrict__ pointwise_bias,
    float* __restrict__ output,
    int B, int IC, int OC,
    int H, int W,
    int KH, int KW,
    int stride, int padding, int dilation,
    int OH, int OW,
    bool use_depthwise_bias, bool use_pointwise_bias)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * OC * OH * OW;
    if (idx >= total_elements) return;

    // Decode linear index to (b, oc, oh, ow)
    int tmp = idx;
    int ow = tmp % OW; tmp /= OW;
    int oh = tmp % OH; tmp /= OH;
    int oc = tmp % OC; tmp /= OC;
    int b = tmp;

    float value = 0.0f;
    for (int ic = 0; ic < IC; ++ic) {
        float dw_sum = 0.0f;
        for (int ky = 0; ky < KH; ++ky) {
            for (int kx = 0; kx < KW; ++kx) {
                int ih = oh * stride - padding + ky * dilation;
                int iw = ow * stride - padding + kx * dilation;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    int input_idx = ((b * IC + ic) * H + ih) * W + iw;
                    int dw_idx = ((ic * 1 + 0) * KH + ky) * KW + kx;
                    dw_sum += input[input_idx] * depthwise_weight[dw_idx];
                }
            }
        }
        if (use_depthwise_bias) {
            dw_sum += depthwise_bias[ic];
        }
        int pw_idx = (oc * IC + ic) * 1 * 1 + 0;  // pointwise weight shape: (OC, IC, 1, 1)
        value += dw_sum * pointwise_weight[pw_idx];
    }
    if (use_pointwise_bias) {
        value += pointwise_bias[oc];
    }
    output[idx] = value;
}

torch::Tensor fused_depthwise_separable_cuda(
    torch::Tensor input,
    torch::Tensor depthwise_weight,
    torch::Tensor pointwise_weight,
    torch::optional<torch::Tensor> depthwise_bias,
    torch::optional<torch::Tensor> pointwise_bias,
    int stride, int padding, int dilation)
{
    int B = input.size(0);
    int IC = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int KH = depthwise_weight.size(2);
    int KW = depthwise_weight.size(3);
    int OC = pointwise_weight.size(0);

    int OH = (H + 2 * padding - dilation * (KH - 1) - 1) / stride + 1;
    int OW = (W + 2 * padding - dilation * (KW - 1) - 1) / stride + 1;

    auto output = torch::zeros({B, OC, OH, OW}, input.options());

    bool use_depthwise_bias = depthwise_bias.has_value();
    bool use_pointwise_bias = pointwise_bias.has_value();
    const float* depthwise_bias_ptr = use_depthwise_bias ? depthwise_bias.value().data_ptr<float>() : nullptr;
    const float* pointwise_bias_ptr = use_pointwise_bias ? pointwise_bias.value().data_ptr<float>() : nullptr;

    int total_elements = B * OC * OH * OW;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_depthwise_separable_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        depthwise_weight.data_ptr<float>(),
        pointwise_weight.data_ptr<float>(),
        depthwise_bias_ptr,
        pointwise_bias_ptr,
        output.data_ptr<float>(),
        B, IC, OC, H, W, KH, KW,
        stride, padding, dilation,
        OH, OW,
        use_depthwise_bias, use_pointwise_bias
    );

    return output;
}
"""

fused_op_cpp_source = "torch::Tensor fused_depthwise_separable_cuda(torch::Tensor input, torch::Tensor depthwise_weight, torch::Tensor pointwise_weight, torch::optional<torch::Tensor> depthwise_bias, torch::optional<torch::Tensor> pointwise_bias, int stride, int padding, int dilation);"

# Compile the custom CUDA operator
fused_op = load_inline(
    name="fused_depthwise_separable",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_depthwise_separable_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Performs a depthwise-separable 2D convolution operation using a fused CUDA kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.has_bias = bias

        # Depthwise filter (in_channels groups, each with 1 input channel)
        self.depthwise_weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size, kernel_size))
        # Pointwise filter (1x1 convolution)
        self.pointwise_weight = nn.Parameter(torch.empty(out_channels, in_channels, 1, 1))

        if bias:
            self.depthwise_bias = nn.Parameter(torch.empty(in_channels))
            self.pointwise_bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('depthwise_bias', None)
            self.register_parameter('pointwise_bias', None)

        self.fused_op = fused_op
        self.reset_parameters()

    def reset_parameters(self):
        # Match standard Conv2d initialization
        nn.init.kaiming_uniform_(self.depthwise_weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.pointwise_weight, a=5**0.5)
        if self.has_bias:
            fan_in_depth = self.kernel_size * self.kernel_size
            bound_depth = 1 / (fan_in_depth ** 0.5)
            nn.init.uniform_(self.depthwise_bias, -bound_depth, bound_depth)
            fan_in_point = self.in_channels
            bound_point = 1 / (fan_in_point ** 0.5)
            nn.init.uniform_(self.pointwise_bias, -bound_point, bound_point)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise-separable 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return self.fused_op.fused_depthwise_separable_cuda(
            x,
            self.depthwise_weight,
            self.pointwise_weight,
            self.depthwise_bias,
            self.pointwise_bias,
            self.stride,
            self.padding,
            self.dilation
        )