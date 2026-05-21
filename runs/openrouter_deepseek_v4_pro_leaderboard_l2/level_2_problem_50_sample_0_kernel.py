import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for fused scale + avg_pool + bias + scale
fused_kernel_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_scale_pool_bias_scale_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ beta,
    float alpha,
    int N, int C, int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_per_batch = C * D_out * H_out * W_out;
    int n = idx / total_per_batch;
    if (n >= N) return;
    int rem = idx % total_per_batch;
    int c = rem / (D_out * H_out * W_out);
    int rem2 = rem % (D_out * H_out * W_out);
    int d_out = rem2 / (H_out * W_out);
    int rem3 = rem2 % (H_out * W_out);
    int h_out = rem3 / W_out;
    int w_out = rem3 % W_out;

    int d_start = d_out * 2;
    int h_start = h_out * 2;
    int w_start = w_out * 2;

    float sum = 0.0f;
    for (int dd = 0; dd < 2; ++dd) {
        for (int hh = 0; hh < 2; ++hh) {
            for (int ww = 0; ww < 2; ++ww) {
                int d = d_start + dd;
                int h = h_start + hh;
                int w = w_start + ww;
                int in_idx = ((n * C + c) * D_in + d) * H_in + h) * W_in + w;
                sum += input[in_idx];
            }
        }
    }
    output[idx] = alpha * sum + beta[c];
}

torch::Tensor fused_scale_pool_bias_scale_cuda(
    torch::Tensor input,
    float scale1,
    torch::Tensor bias,
    float scale2)
{
    input = input.contiguous();
    auto N = input.size(0);
    auto C = input.size(1);
    auto D_in = input.size(2);
    auto H_in = input.size(3);
    auto W_in = input.size(4);
    int D_out = D_in / 2;
    int H_out = H_in / 2;
    int W_out = W_in / 2;
    auto output = torch::empty({N, C, D_out, H_out, W_out}, input.options());

    auto beta = (bias * scale2).view({C}).contiguous();
    float alpha = scale1 * scale2 / 8.0f;

    int total_elements = N * C * D_out * H_out * W_out;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_scale_pool_bias_scale_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        beta.data_ptr<float>(),
        alpha,
        N, C, D_in, H_in, W_in,
        D_out, H_out, W_out);

    return output;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_scale_pool_bias_scale_cuda(
    torch::Tensor input,
    float scale1,
    torch::Tensor bias,
    float scale2);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_scale_pool_bias_scale",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_cuda_source,
    functions=["fused_scale_pool_bias_scale_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale1 = nn.Parameter(torch.tensor(scale1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale2 = nn.Parameter(torch.tensor(scale2))
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv_transpose(x)
        # Fused scale1, avg_pool, bias, scale2
        x = self.fused_op.fused_scale_pool_bias_scale_cuda(x, self.scale1.item(), self.bias, self.scale2.item())
        return x