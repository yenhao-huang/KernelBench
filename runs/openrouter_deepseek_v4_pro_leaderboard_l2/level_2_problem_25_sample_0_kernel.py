import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for convolution + min + tanh + tanh fusion
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_min_tanh_tanh_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int out_height,
    int out_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_height * out_width;
    
    if (idx < total_elements) {
        int b = idx / (out_height * out_width);
        int hw = idx % (out_height * out_width);
        int h = hw / out_width;
        int w = hw % out_width;
        
        // Compute convolution for all output channels at this spatial position
        // and find the minimum across channels
        float min_val = INFINITY;
        float conv_results[64]; // Assuming max out_channels = 64, adjust if needed
        
        for (int oc = 0; oc < out_channels; oc++) {
            float sum = bias[oc];
            for (int ic = 0; ic < in_channels; ic++) {
                for (int kh = 0; kh < kernel_size; kh++) {
                    for (int kw = 0; kw < kernel_size; kw++) {
                        int ih = h + kh;
                        int iw = w + kw;
                        sum += input[b * in_channels * height * width + ic * height * width + ih * width + iw] *
                               weight[oc * in_channels * kernel_size * kernel_size + ic * kernel_size * kernel_size + kh * kernel_size + kw];
                    }
                }
            }
            conv_results[oc] = sum;
            if (sum < min_val) {
                min_val = sum;
            }
        }
        
        // Apply tanh twice to the minimum value
        float tanh1 = tanhf(min_val);
        float tanh2 = tanhf(tanh1);
        
        // Write result (broadcast to all channels? The original min keeps dim=1, so output has 1 channel)
        output[b * out_height * out_width + h * out_width + w] = tanh2;
    }
}

torch::Tensor fused_conv_min_tanh_tanh_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int out_channels = weight.size(0);
    int out_height = height - kernel_size + 1;
    int out_width = width - kernel_size + 1;
    
    auto output = torch::empty({batch_size, 1, out_height, out_width}, input.options());
    
    const int block_size = 256;
    const int num_blocks = (batch_size * out_height * out_width + block_size - 1) / block_size;
    
    fused_conv_min_tanh_tanh_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        out_height,
        out_width
    );
    
    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_conv_min_tanh_tanh_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int kernel_size);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_conv_min_tanh_tanh_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.fused_ops = fused_ops

    def forward(self, x):
        # Use the fused kernel that does conv + min + tanh + tanh
        return self.fused_ops.fused_conv_min_tanh_tanh_cuda(
            x, self.conv.weight, self.conv.bias, self.conv.kernel_size[0]
        )