import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for fused average pooling (kernel_size=4, stride=4)
fused_avg_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_avg_pool3d_kernel(const float* input, float* output,
                                        int N, int C, int D, int H, int W,
                                        int D_out, int H_out, int W_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * D_out * H_out * W_out;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int d_out = (idx / (W_out * H_out)) % D_out;
    int c = (idx / (W_out * H_out * D_out)) % C;
    int n = idx / (W_out * H_out * D_out * C);

    int d_start = d_out * 4;
    int h_start = h_out * 4;
    int w_start = w_out * 4;

    float sum = 0.0f;
    for (int dd = 0; dd < 4; ++dd) {
        for (int hh = 0; hh < 4; ++hh) {
            for (int ww = 0; ww < 4; ++ww) {
                int d = d_start + dd;
                int h = h_start + hh;
                int w = w_start + ww;
                sum += input[((n * C + c) * D + d) * H * W + h * W + w];
            }
        }
    }
    output[idx] = sum / 64.0f;
}

torch::Tensor fused_avg_pool3d_cuda(torch::Tensor input) {
    TORCH_CHECK(input.dim() == 5, "Input must be 5D");
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    int N = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    int D_out = D / 4;
    int H_out = H / 4;
    int W_out = W / 4;

    auto output = torch::empty({N, C, D_out, H_out, W_out}, input.options());

    int total_elements = N * C * D_out * H_out * W_out;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_avg_pool3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        N, C, D, H, W, D_out, H_out, W_out);

    return output;
}
"""

fused_avg_pool_cpp_source = "torch::Tensor fused_avg_pool3d_cuda(torch::Tensor input);"

# Compile the custom CUDA operator
fused_avg_pool = load_inline(
    name="fused_avg_pool",
    cpp_sources=fused_avg_pool_cpp_source,
    cuda_sources=fused_avg_pool_source,
    functions=["fused_avg_pool3d_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = x.contiguous()
        x = fused_avg_pool.fused_avg_pool3d_cuda(x)
        return x