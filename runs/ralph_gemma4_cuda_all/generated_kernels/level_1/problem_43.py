import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# CUDA implementation of MaxPool3d
# This kernel uses a simple parallelization strategy where each thread handles one output element.
# It accounts for padding, dilation, stride, and ceil_mode.
maxpool3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

__global__ void maxpool3d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels, int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size, int stride, int padding, int dilation) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * out_d * out_h * out_w;

    if (idx < total_elements) {
        // Decompose idx into output coordinates
        int w_out = idx % out_w;
        int remaining = idx / out_w;
        int h_out = remaining % out_h;
        remaining /= out_h;
        int d_out = remaining % out_d;
        remaining /= out_d;
        int c_out = remaining % channels;
        int b_out = remaining / channels;

        // Calculate input base coordinates
        // input_idx = b*C*D*H*W + c*D*H*W + d*H*W + h*W + w
        int in_base_idx = (((b_out * channels + c_out) * in_d + 0) * in_h + 0) * in_w; // placeholder logic

        float max_val = -std::numeric_limits<float>::infinity();

        for (int kd = 0; kd < kernel_size; ++kd) {
            int d_in = d_out * stride - padding + kd * dilation;
            for (int kh = 0; kh < kernel_size; ++kh) {
                int h_in = h_out * stride - padding + kh * dilation;
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int w_in = w_out * stride - padding + kw * dilation;

                    if (d_in >= 0 && d_in < in_d && h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
                        float val = input[(((b_out * channels + c_out) * in_d + d_in) * in_h + h_in) * in_w + w_in];
                        if (val > max_val) {
                            max_val = val;
                        }
                    }
                }
            }
        }
        output[idx] = max_val;
    }
}

torch::Tensor maxpool3d_cuda(
    torch::Tensor input,
    int kernel_size, int stride, int padding, int dilation,
    int out_d, int out_h, int out_w) 
{
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto in_d = input.size(2);
    auto in_h = input.size(3);
    auto in_w = input.size(4);

    auto output = torch::empty({batch_size, channels, out_d, out_h, out_w}, input.options());

    int total_elements = output.numel();
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    maxpool3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_size, stride, padding, dilation
    );

    return output;
}
"""

maxpool3d_cpp_source = """
torch::Tensor maxpool3d_cuda(
    torch::Tensor input,
    int kernel_size, int stride, int padding, int dilation,
    int out_d, int out_h, int out_w);
"""

# Compile the inline CUDA code
maxpool3d_lib = load_inline(
    name="maxpool3d_lib",
    cpp_sources=maxpool3d_cpp_source,
    cuda_sources=maxpool3d_source,
    functions=["maxpool3d_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        self.maxpool_lib = maxpool3d_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            # Fallback for CPU
            return nn.MaxPool3d(
                kernel_size=self.kernel_size, 
                stride=self.stride, 
                padding=self.padding, 
                dilation=self.dilation, 
                return_indices=self.return_indices, 
                ceil_mode=self.ceil_mode
            )(x)

        batch_size, channels, in_d, in_h, in_w = x.shape
        
        # Calculate output dimensions
        def get_out_dim(in_size, kernel, stride, padding, dilation, ceil_mode):
            if ceil_mode:
                return math.ceil((in_size + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)
            else:
                return math.floor((in_size + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)

        out_d = get_out_dim(in_d, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode)
        out_h = get_out_dim(in_h, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode)
        out_w = get_out_dim(in_w, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode)

        # Note: Our custom kernel currently only supports return_indices=False
        if self.return_indices:
            return nn.MaxPool3d(
                kernel_size=self.kernel_size, 
                stride=self.stride, 
                padding=self.padding, 
                dilation=self.dilation, 
                return_indices=True, 
                ceil_mode=self.ceil_mode
            )(x)

        return self.maxpool_lib.maxpool3d_cuda(
            x.contiguous(),
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            out_d,
            out_h,
            out_w
        )