import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for Depthwise Convolution
# This kernel fuses the depthwise convolution operation.
# It handles asymmetric kernels, strides, padding, and dilation.
depthwise_conv_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void depthwise_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int in_h, int in_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int out_h, int out_w,
    bool has_bias) 
{
    // Each thread handles one output element (batch, channel, oh, ow)
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ch = blockIdx.z % in_channels;
    int b = blockIdx.z / in_channels;

    if (oh < out_h && ow < out_w) {
        float sum = 0.0f;
        
        // Calculate the top-left position in the input tensor
        int ih_start = oh * stride_h - pad_h;
        int iw_start = ow * stride_w - pad_w;

        for (int kh = 0; kh < kernel_h; ++kh) {
            int ih = ih_start + kh * dilation_h;
            if (ih >= 0 && ih < in_h) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int iw = iw_start + kw * dilation_w;
                    if (iw >= 0 && iw < in_w) {
                        // input index: b * (C*H*W) + ch * (H*W) + ih * W + iw
                        int input_idx = ((b * in_channels + ch) * in_h + ih) * in_w + iw;
                        // weight index: ch * (kH*kW) + kh * kW + kw
                        int weight_idx = ch * (kernel_h * kernel_w) + kh * kernel_w + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }

        if (has_bias) {
            sum += bias[ch];
        }

        // output index: b * (C*OH*OW) + ch * (OH*OW) + oh * OW + ow
        int output_idx = ((b * in_channels + ch) * out_h + oh) * out_w + ow;
        output[output_idx] = sum;
    }
}

torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    at::optional<torch::Tensor> bias,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w) 
{
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_h = input.size(2);
    const int in_w = input.size(3);

    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);

    const int out_h = (in_h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int out_w = (in_w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    auto output = torch::empty({batch_size, in_channels, out_h, out_w}, input.options());
    
    dim3 block_size(16, 16, 1);
    dim3 grid_size((out_w + block_size.x - 1) / block_size.x, 
                   (out_h + block_size.y - 1) / block_size.y, 
                   batch_size * in_channels);

    bool has_bias = bias.has_value();
    const float* bias_ptr = has_bias ? bias.value().data_ptr<float>() : nullptr;

    depthwise_conv2d_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size, in_channels, in_h, in_w,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        out_h, out_w,
        has_bias
    );

    return output;
}
"""

depthwise_conv_cpp_source = """
torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    at::optional<torch::Tensor> bias,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w);
"""

# Compile the inline CUDA code
depthwise_conv_lib = load_inline(
    name="depthwise_conv_lib",
    cpp_sources=depthwise_conv_cpp_source,
    cuda_sources=depthwise_conv_cuda_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, 
                 stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, 
                 dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        
        # We still use nn.Parameter to manage weights and biases easily
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size_h, kernel_size_w))
        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels))
        else:
            self.register_parameter('bias', None)
            
        self.depthwise_conv_lib = depthwise_conv_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous for the CUDA kernel
        x = x.contiguous()
        weight = self.weight.contiguous()
        
        bias_opt = self.bias if self.bias is not None else None
        if bias_opt is not None:
            bias_opt = bias_opt.contiguous()

        return self.depthwise_conv_lib.depthwise_conv2d_cuda(
            x,
            weight,
            bias_opt,
            self.stride_h,
            self.stride_w,
            self.padding_h,
            self.padding_w,
            self.dilation_h,
            self.dilation_w
        )