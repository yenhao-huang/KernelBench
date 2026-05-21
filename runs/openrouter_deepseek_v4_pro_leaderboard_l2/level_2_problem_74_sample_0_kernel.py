import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for fused LeakyReLU + multiply + LeakyReLU + maxpool3d
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_leaky_mul_leaky_maxpool3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ multiplier,
    float* __restrict__ output,
    int N, int C, int D, int H, int W,
    int D_out, int H_out, int W_out,
    float negative_slope)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = N * C * D_out * H_out * W_out;
    if (idx >= total_output_elements) return;

    // Compute output indices
    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int d_out = (idx / (W_out * H_out)) % D_out;
    int c = (idx / (W_out * H_out * D_out)) % C;
    int n = idx / (W_out * H_out * D_out * C);

    // Input window base indices
    int d_base = d_out * 2;
    int h_base = h_out * 2;
    int w_base = w_out * 2;

    float max_val = -1e30f;
    for (int dd = 0; dd < 2; ++dd) {
        int d_idx = d_base + dd;
        if (d_idx >= D) continue;
        for (int hh = 0; hh < 2; ++hh) {
            int h_idx = h_base + hh;
            if (h_idx >= H) continue;
            for (int ww = 0; ww < 2; ++ww) {
                int w_idx = w_base + ww;
                if (w_idx >= W) continue;
                int input_idx = ((n * C + c) * D + d_idx) * H * W + h_idx * W + w_idx;
                float val = input[input_idx];
                // First LeakyReLU
                val = val > 0.0f ? val : negative_slope * val;
                // Multiply by per-channel multiplier
                val = val * multiplier[c];
                // Second LeakyReLU
                val = val > 0.0f ? val : negative_slope * val;
                if (val > max_val) max_val = val;
            }
        }
    }
    output[idx] = max_val;
}

torch::Tensor fused_leaky_mul_leaky_maxpool3d_cuda(
    torch::Tensor input,
    torch::Tensor multiplier,
    float negative_slope)
{
    // input: (N, C, D, H, W)
    auto N = input.size(0);
    auto C = input.size(1);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);

    int D_out = D / 2;
    int H_out = H / 2;
    int W_out = W / 2;

    auto output = torch::empty({N, C, D_out, H_out, W_out}, input.options());

    int total_elements = N * C * D_out * H_out * W_out;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    // Ensure multiplier is contiguous and 1D
    auto multiplier_flat = multiplier.contiguous().view({-1});

    fused_leaky_mul_leaky_maxpool3d_kernel<<<num_blocks, block_size>>>(
        input.contiguous().data_ptr<float>(),
        multiplier_flat.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D, H, W,
        D_out, H_out, W_out,
        negative_slope);

    return output;
}
"""

fused_op_cpp_source = """
torch::Tensor fused_leaky_mul_leaky_maxpool3d_cuda(
    torch::Tensor input,
    torch::Tensor multiplier,
    float negative_slope);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_leaky_mul_leaky_maxpool3d",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_leaky_mul_leaky_maxpool3d_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model: ConvTranspose3d unchanged, then a single fused CUDA kernel
    performs LeakyReLU -> multiply by learnable parameter -> LeakyReLU -> MaxPool3d.
    This reduces kernel launches and memory bandwidth.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.negative_slope = 0.2
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv_transpose(x)
        # Fused LeakyReLU + multiply + LeakyReLU + MaxPool3d(kernel_size=2, stride=2)
        x = self.fused_op.fused_leaky_mul_leaky_maxpool3d_cuda(
            x, self.multiplier, self.negative_slope
        )
        return x