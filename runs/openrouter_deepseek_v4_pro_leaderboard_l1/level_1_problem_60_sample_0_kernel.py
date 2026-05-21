import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA source for 3D convolution
conv3d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const float* __restrict__ bias,
    int B, int IC, int D, int H, int W,
    int OC, int KD, int KH, int KW,
    int OD, int OH, int OW,
    int stride, int padding, int dilation, int groups,
    bool has_bias) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * OC * OD * OH * OW;
    if (idx >= total) return;

    int ow = idx % OW;
    int oh = (idx / OW) % OH;
    int od = (idx / (OW * OH)) % OD;
    int oc = (idx / (OW * OH * OD)) % OC;
    int b = idx / (OW * OH * OD * OC);

    int IC_per_group = IC / groups;
    int OC_per_group = OC / groups;
    int group = oc / OC_per_group;
    int oc_in_group = oc % OC_per_group;
    int ic_start = group * IC_per_group;
    int ic_end = ic_start + IC_per_group;

    float sum = 0.0f;
    for (int ic = ic_start; ic < ic_end; ++ic) {
        for (int kd = 0; kd < KD; ++kd) {
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    int in_d = od * stride + kd * dilation - padding;
                    int in_h = oh * stride + kh * dilation - padding;
                    int in_w = ow * stride + kw * dilation - padding;
                    if (in_d >= 0 && in_d < D && in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
                        float input_val = input[((b * IC + ic) * D + in_d) * H * W + in_h * W + in_w];
                        float weight_val = weight[((oc * IC_per_group + (ic - ic_start)) * KD + kd) * KH * KW + kh * KW + kw];
                        sum += input_val * weight_val;
                    }
                }
            }
        }
    }
    if (has_bias) {
        sum += bias[oc];
    }
    output[idx] = sum;
}

torch::Tensor conv3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride, int padding, int dilation, int groups) {
    
    const auto B = input.size(0);
    const auto IC = input.size(1);
    const auto D = input.size(2);
    const auto H = input.size(3);
    const auto W = input.size(4);
    const auto OC = weight.size(0);
    const auto KD = weight.size(2);
    const auto KH = weight.size(3);
    const auto KW = weight.size(4);
    
    auto OD = (D + 2 * padding - dilation * (KD - 1) - 1) / stride + 1;
    auto OH = (H + 2 * padding - dilation * (KH - 1) - 1) / stride + 1;
    auto OW = (W + 2 * padding - dilation * (KW - 1) - 1) / stride + 1;
    
    auto output = torch::empty({B, OC, OD, OH, OW}, input.options());
    
    int total = B * OC * OD * OH * OW;
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;
    
    bool has_bias = bias.has_value();
    const float* bias_ptr = has_bias ? bias.value().data_ptr<float>() : nullptr;
    
    conv3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        bias_ptr,
        B, IC, D, H, W,
        OC, KD, KH, KW,
        OD, OH, OW,
        stride, padding, dilation, groups,
        has_bias);
    
    return output;
}
"""

conv3d_cpp_source = "torch::Tensor conv3d_cuda(torch::Tensor input, torch::Tensor weight, torch::optional<torch::Tensor> bias, int stride, int padding, int dilation, int groups);"

# Compile the inline CUDA code
conv3d_op = load_inline(
    name="conv3d_cuda_op",
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_cuda_source,
    functions=["conv3d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Performs a standard 3D convolution operation using a custom CUDA kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (depth, height, width).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding applied to the input. Defaults to 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size  # (D, H, W)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        KD, KH, KW = kernel_size
        weight_shape = (out_channels, in_channels // groups, KD, KH, KW)
        self.weight = nn.Parameter(torch.empty(*weight_shape))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self._init_parameters()
        self.custom_conv = conv3d_op
        
    def _init_parameters(self):
        # Mimic default Conv3d initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution using the custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        return self.custom_conv.conv3d_cuda(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)