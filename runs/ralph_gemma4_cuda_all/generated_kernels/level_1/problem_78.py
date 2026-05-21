import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for Transposed Convolution (simplified version for demonstration)
# In a real-world scenario, a production-grade ConvTranspose2d kernel would use 
# highly optimized tiling and shared memory strategies (like cuDNN).
# Here we implement a fused kernel that handles the accumulation logic.

transposed_conv_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void transposed_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int in_h, int in_w, int out_h, int out_w,
    int k_h, int k_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w) 
{
    // Using a simple atomicAdd approach for the transposed convolution logic
    // Transposed convolution is essentially the gradient of a standard convolution.
    // For each input pixel, we scatter the weighted kernel onto the output.
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * in_channels * in_h * in_w;

    if (idx < total_elements) {
        // Decode input index
        int w_in = idx % in_w;
        int h_in = (idx / in_w) % in_h;
        int c_in = (idx / (in_w * in_h)) % in_channels;
        int b = idx / (in_w * in_h * in_channels);

        float val = input[idx];

        for (int oc = 0; oc < out_channels; ++oc) {
            for (int kh = 0; kh < k_h; ++kh) {
                for (int kw = 0; kw < k_w; ++kw) {
                    int oh = h_in * stride_h - pad_h + kh;
                    int ow = w_in * stride_w - pad_w + kw;

                    if (oh >= 0 && oh < out_h && ow >= 0 && ow < out_w) {
                        // weight shape: [in_channels, out_channels, k_h, k_w]
                        // Note: PyTorch ConvTranspose2d weight is [in_channels, out_channels, k_h, k_w]
                        int weight_idx = ((c_in * out_channels + oc) * k_h + kh) * k_w + kw;
                        int out_idx = (((b * out_channels + oc) * out_h + oh) * out_w + ow);
                        
                        atomicAdd(&output[out_idx], val * weight[weight_idx]);
                    }
                }
            }
        }
    }
}

// Kernel to add bias
__global__ void add_bias_kernel(float* output, const float* bias, int batch_size, int out_channels, int out_h, int out_w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_h * out_w;
    if (idx < total_elements) {
        int w = idx % out_w;
        int h = (idx / out_w) % out_h;
        int c = (idx / (out_w * out_h)) % out_channels;
        output[idx] += bias[c];
    }
}

torch::Tensor transposed_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    at::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> kernel_size) 
{
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_h = input.size(2);
    const int in_w = input.size(3);
    
    const int out_channels = weight.size(1);
    const int k_h = kernel_size[0];
    const int k_w = kernel_size[1];
    
    const int stride_h = stride[0];
    const int stride_w = stride[1];
    const int pad_h = padding[0];
    const int pad_w = padding[1];

    // Calculate output dimensions
    int out_h = (in_h - 1) * stride_h - 2 * pad_h + k_h;
    int out_w = (in_w - 1) * stride_w - 2 * pad_w + k_w;

    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, input.options());

    int total_input_elements = input.numel();
    int threads = 256;
    int blocks = (total_input_elements + threads - 1) / threads;

    transposed_conv2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_h, in_w, out_h, out_w,
        k_h, k_w,
        stride_h, stride_w,
        pad_h, pad_w
    );

    if (bias.has_value()) {
        int total_out_elements = output.numel();
        int bias_blocks = (total_out_elements + threads - 1) / threads;
        add_bias_kernel<<<bias_blocks, threads>>>(
            output.data_ptr<float>(),
            bias.value().data_ptr<float>(),
            batch_size, out_channels, out_h, out_w
        );
    }

    return output;
}
"""

transposed_conv_cpp_source = """
torch::Tensor transposed_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    at::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> kernel_size);
"""

# Compile the inline CUDA code
transposed_conv_lib = load_inline(
    name="transposed_conv_lib",
    cpp_sources=transposed_conv_cpp_source,
    cuda_sources=transposed_conv_cuda_source,
    functions=["transposed_conv2d_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # We still need the weight and bias parameters managed by PyTorch
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.cuda_op = transposed_conv_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are float32 and on CUDA
        x = x.float().cuda()
        weight = self.weight.float().cuda()
        bias = self.bias.float().cuda() if self.bias is not None else None
        
        # Call the custom CUDA operator
        return self.cuda_op.transposed_conv2d_cuda(
            x, 
            weight, 
            bias, 
            list(self.stride), 
            list(self.padding), 
            list(self.kernel_size)
        )