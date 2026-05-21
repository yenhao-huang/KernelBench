import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for transposed 1D convolution
conv_transpose1d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose1d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int length_in,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int length_out)
{
    int total_elements = batch_size * out_channels * length_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int tid = idx; tid < total_elements; tid += gridDim.x * blockDim.x) {
        int n = tid / (out_channels * length_out);
        int rem = tid % (out_channels * length_out);
        int oc = rem / length_out;
        int pos = rem % length_out;
        
        float sum = bias != nullptr ? bias[oc] : 0.0f;
        
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int k = 0; k < kernel_size; ++k) {
                int diff = pos + 2 * padding - dilation * k;
                int input_idx = diff / stride;
                if (diff % stride == 0 && input_idx >= 0 && input_idx < length_in) {
                    // weight layout: (in_channels, out_channels, kernel_size)
                    int w_idx = ic * out_channels * kernel_size + oc * kernel_size + k;
                    int a_idx = n * in_channels * length_in + ic * length_in + input_idx;
                    sum += weight[w_idx] * input[a_idx];
                }
            }
        }
        output[tid] = sum;
    }
}

torch::Tensor conv_transpose1d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation)
{
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int length_in = input.size(2);
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);
    
    int length_out = (length_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    
    auto output = torch::empty({batch_size, out_channels, length_out}, input.options());
    
    const dim3 block_size(256);
    const dim3 num_blocks((batch_size * out_channels * length_out + block_size.x - 1) / block_size.x);
    
    conv_transpose1d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        (bias.defined() ? bias.data_ptr<float>() : nullptr),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        length_in,
        kernel_size,
        stride,
        padding,
        dilation,
        length_out
    );
    
    return output;
}
"""

conv_transpose1d_cpp_source = """
torch::Tensor conv_transpose1d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation);
"""

# Compile the custom CUDA operator
conv_transpose1d_op = load_inline(
    name="conv_transpose1d_cuda_extension",
    cpp_sources=conv_transpose1d_cpp_source,
    cuda_sources=conv_transpose1d_cuda_source,
    functions=["conv_transpose1d_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Keep the original ConvTranspose1d layer to hold the learnable parameters
        self.conv1d_transpose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size,
                                                   stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.op = conv_transpose1d_op
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use the custom CUDA kernel for the forward pass
        bias = self.conv1d_transpose.bias if self.conv1d_transpose.bias is not None else torch.Tensor()
        return self.op.conv_transpose1d_cuda(
            x.contiguous(),
            self.conv1d_transpose.weight.contiguous(),
            bias,
            self.stride,
            self.padding,
            self.dilation
        )