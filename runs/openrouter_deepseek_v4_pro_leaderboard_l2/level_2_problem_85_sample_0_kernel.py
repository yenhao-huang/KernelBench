import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused maxpool + scale + clamp
maxpool_scale_clamp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void maxpool_scale_clamp_kernel(
    const float* __restrict__ input,
    const float* __restrict__ scale,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int kernel_size, int stride, int padding,
    float clamp_min, float clamp_max) {

    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int batch_channel_idx = blockIdx.z;
    int n = batch_channel_idx / C;
    int c = batch_channel_idx % C;

    if (n < N && c < C && h_out < H_out && w_out < W_out) {
        int h_in_start = h_out * stride - padding;
        int w_in_start = w_out * stride - padding;

        float max_val = -1e38f;
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int h_in = h_in_start + kh;
                int w_in = w_in_start + kw;
                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    float val = input[((n * C + c) * H + h_in) * W + w_in];
                    if (val > max_val) max_val = val;
                }
            }
        }

        max_val *= scale[c];
        max_val = fminf(fmaxf(max_val, clamp_min), clamp_max);

        output[((n * C + c) * H_out + h_out) * W_out + w_out] = max_val;
    }
}

torch::Tensor maxpool_scale_clamp_cuda(
    torch::Tensor input,
    torch::Tensor scale,
    int kernel_size,
    float clamp_min,
    float clamp_max) {

    auto N = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);

    int stride = kernel_size;
    int padding = 0;

    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({N, C, H_out, W_out}, input.options());

    const int block_x = 16;
    const int block_y = 16;
    dim3 block(block_x, block_y);
    dim3 grid(
        (W_out + block_x - 1) / block_x,
        (H_out + block_y - 1) / block_y,
        N * C
    );

    auto scale_contig = scale.contiguous().view({C});

    maxpool_scale_clamp_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        scale_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        H_out, W_out,
        kernel_size, stride, padding,
        clamp_min, clamp_max
    );

    return output;
}
"""

maxpool_scale_clamp_cpp_source = (
    "torch::Tensor maxpool_scale_clamp_cuda(torch::Tensor input, torch::Tensor scale, int kernel_size, float clamp_min, float clamp_max);"
)

# Compile the inline CUDA code
maxpool_scale_clamp = load_inline(
    name="maxpool_scale_clamp",
    cpp_sources=maxpool_scale_clamp_cpp_source,
    cuda_sources=maxpool_scale_clamp_source,
    functions=["maxpool_scale_clamp_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model with fused maxpool + scale + clamp CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.fused_op = maxpool_scale_clamp

    def forward(self, x):
        x = self.conv(x)
        x = self.group_norm(x)
        # Fused maxpool, scale multiplication, and clamp
        x = self.fused_op.maxpool_scale_clamp_cuda(
            x, self.scale, self.maxpool_kernel_size, self.clamp_min, self.clamp_max
        )
        return x


# The following functions remain unchanged
batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128 
kernel_size = 3
num_groups = 16
scale_shape = (out_channels, 1, 1)
maxpool_kernel_size = 4
clamp_min = 0.0
clamp_max = 1.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max]