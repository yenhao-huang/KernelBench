import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA source for custom 3D convolution
conv3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv3d_kernel(
    const float* input, const float* weight, const float* bias,
    float* output,
    int batch, int in_channels, int out_channels,
    int depth, int width, int height,
    int kernel_d, int kernel_w, int kernel_h,
    int stride, int dilation, int groups,
    int out_depth, int out_width, int out_height, bool has_bias) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_channels * out_depth * out_width * out_height;
    if (idx >= total) return;
    
    // Decompose linear index into batch, channel, depth, width, height
    int tmp = idx;
    int oh = tmp % out_height; tmp /= out_height;
    int ow = tmp % out_width; tmp /= out_width;
    int od = tmp % out_depth; tmp /= out_depth;
    int oc = tmp % out_channels; tmp /= out_channels;
    int n = tmp;

    int in_channels_per_group = in_channels / groups;
    int out_channels_per_group = out_channels / groups;
    int group = oc / out_channels_per_group;
    int in_channel_start = group * in_channels_per_group;
    int in_channel_end = (group + 1) * in_channels_per_group;

    float value = 0.0f;
    // Compute starting positions in input for the current output voxel
    int id_start = od * stride;
    int iw_start = ow * stride;
    int ih_start = oh * stride;

    for (int ic = in_channel_start; ic < in_channel_end; ++ic) {
        for (int kd = 0; kd < kernel_d; ++kd) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                for (int kh = 0; kh < kernel_h; ++kh) {
                    int id = id_start + kd * dilation;
                    int iw = iw_start + kw * dilation;
                    int ih = ih_start + kh * dilation;
                    // Input layout: (batch, in_channels, depth, width, height)
                    int input_index = ((((n * in_channels + ic) * depth + id) * width + iw) * height + ih);
                    int weight_ic = ic - in_channel_start;
                    // Weight layout: (out_channels, in_channels_per_group, D, W, H)
                    int weight_index = ((((oc * in_channels_per_group + weight_ic) * kernel_d + kd) * kernel_w + kw) * kernel_h + kh);
                    value += input[input_index] * weight[weight_index];
                }
            }
        }
    }
    if (has_bias) {
        value += bias[oc];
    }
    output[idx] = value;
}

torch::Tensor conv3d_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int stride, int padding, int dilation, int groups, bool has_bias) {
    
    // Ensure input is contiguous for direct pointer access
    input = input.contiguous();
    weight = weight.contiguous();
    if (has_bias) bias = bias.contiguous();
    
    int batch = input.size(0);
    int in_channels = input.size(1);
    int depth = input.size(2);
    int width = input.size(3);
    int height = input.size(4);
    int out_channels = weight.size(0);
    int kernel_d = weight.size(2);
    int kernel_w = weight.size(3);
    int kernel_h = weight.size(4);
    
    // Compute output dimensions from padded input
    int out_depth = (depth - dilation * (kernel_d - 1) - 1) / stride + 1;
    int out_width = (width - dilation * (kernel_w - 1) - 1) / stride + 1;
    int out_height = (height - dilation * (kernel_h - 1) - 1) / stride + 1;
    
    auto output = torch::zeros({batch, out_channels, out_depth, out_width, out_height}, input.options());
    
    int total = batch * out_channels * out_depth * out_width * out_height;
    int block = 256;
    int grid = (total + block - 1) / block;
    
    conv3d_kernel<<<grid, block>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        has_bias ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch, in_channels, out_channels,
        depth, width, height,
        kernel_d, kernel_w, kernel_h,
        stride, dilation, groups,
        out_depth, out_width, out_height, has_bias
    );
    
    return output;
}
"""

conv3d_cpp_source = "torch::Tensor conv3d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride, int padding, int dilation, int groups, bool has_bias);"

# Compile the inline CUDA code
custom_conv3d = load_inline(
    name="custom_conv3d",
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_source,
    functions=["conv3d_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized 3D convolution using custom CUDA kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.has_bias = bias
        
        # Create weight and optional bias parameters
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters (default initialization similar to Conv3d)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
        self.custom_conv3d = custom_conv3d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply 3D constant zero padding if needed
        if self.padding > 0:
            pad_tup = (self.padding, self.padding, self.padding, self.padding, self.padding, self.padding)
            x = F.pad(x, pad_tup, "constant", 0)
        
        # Ensure input is contiguous (custom kernel expects contiguous memory)
        x = x.contiguous()
        weight = self.weight.contiguous()
        bias = self.bias.contiguous() if self.has_bias else torch.empty(0, device=x.device)
        
        return self.custom_conv3d.conv3d_cuda(
            x, weight, bias,
            self.stride, self.padding, self.dilation, self.groups, self.has_bias
        )