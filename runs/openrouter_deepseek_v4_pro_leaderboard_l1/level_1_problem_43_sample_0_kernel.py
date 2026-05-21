import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for custom 3D max pooling
max_pool3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void max_pool3d_kernel(const float* input, float* output,
                                  int batch, int channels,
                                  int input_d, int input_h, int input_w,
                                  int output_d, int output_h, int output_w,
                                  int kernel_d, int kernel_h, int kernel_w,
                                  int stride_d, int stride_h, int stride_w,
                                  int padding_d, int padding_h, int padding_w,
                                  int dilation_d, int dilation_h, int dilation_w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch * channels * output_d * output_h * output_w;
    if (idx >= total_elements) return;

    int w = idx % output_w;
    int h = (idx / output_w) % output_h;
    int d = (idx / (output_w * output_h)) % output_d;
    int c = (idx / (output_w * output_h * output_d)) % channels;
    int n = idx / (output_w * output_h * output_d * channels);

    float max_val = -FLT_MAX;
    for (int kd = 0; kd < kernel_d; ++kd) {
        int in_d = d * stride_d - padding_d + kd * dilation_d;
        if (in_d < 0 || in_d >= input_d) continue;
        for (int kh = 0; kh < kernel_h; ++kh) {
            int in_h = h * stride_h - padding_h + kh * dilation_h;
            if (in_h < 0 || in_h >= input_h) continue;
            for (int kw = 0; kw < kernel_w; ++kw) {
                int in_w = w * stride_w - padding_w + kw * dilation_w;
                if (in_w < 0 || in_w >= input_w) continue;
                int input_idx = ((n * channels + c) * input_d + in_d) * input_h + in_h) * input_w + in_w;
                float val = input[input_idx];
                if (val > max_val) max_val = val;
            }
        }
    }
    output[idx] = max_val;
}

torch::Tensor max_pool3d_cuda(torch::Tensor input, int kernel_size, int stride, int padding, int dilation) {
    int batch = input.size(0);
    int channels = input.size(1);
    int input_d = input.size(2);
    int input_h = input.size(3);
    int input_w = input.size(4);

    int kernel_d = kernel_size;
    int kernel_h = kernel_size;
    int kernel_w = kernel_size;
    int stride_d = stride;
    int stride_h = stride;
    int stride_w = stride;
    int padding_d = padding;
    int padding_h = padding;
    int padding_w = padding;
    int dilation_d = dilation;
    int dilation_h = dilation;
    int dilation_w = dilation;

    int output_d = (input_d + 2 * padding_d - dilation_d * (kernel_d - 1) - 1) / stride_d + 1;
    int output_h = (input_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int output_w = (input_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    auto output = torch::empty({batch, channels, output_d, output_h, output_w}, input.options());

    int total_elements = batch * channels * output_d * output_h * output_w;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    max_pool3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        batch, channels,
        input_d, input_h, input_w,
        output_d, output_h, output_w,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        dilation_d, dilation_h, dilation_w
    );

    return output;
}
"""

max_pool3d_cpp_source = "torch::Tensor max_pool3d_cuda(torch::Tensor input, int kernel_size, int stride, int padding, int dilation);"

# Compile the custom CUDA operator
max_pool3d_custom = load_inline(
    name="max_pool3d_custom",
    cpp_sources=max_pool3d_cpp_source,
    cuda_sources=max_pool3d_source,
    functions=["max_pool3d_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        self.max_pool3d_custom = max_pool3d_custom

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.max_pool3d_custom.max_pool3d_cuda(x, self.kernel_size, self.stride, self.padding, self.dilation)