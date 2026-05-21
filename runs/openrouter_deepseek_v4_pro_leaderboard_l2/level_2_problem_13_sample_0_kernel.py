import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source code for the fused kernel
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_mean_bias_softmax_tanh_scale_kernel(
    const float* __restrict__ input,   // [B, C, D, H, W]
    const float* __restrict__ bias,    // [C]
    float* __restrict__ output,        // [B, C, 1, H, W]
    int B, int C, int D, int H, int W,
    float scaling_factor)
{
    int h = blockIdx.x;
    int w = blockIdx.y;
    int b = blockIdx.z;
    int c = threadIdx.x;

    if (h >= H || w >= W || b >= B || c >= C) return;

    // 1. Mean over depth D
    float sum = 0.0f;
    for (int d = 0; d < D; ++d) {
        int idx = (((b * C + c) * D) + d) * H * W + h * W + w;
        sum += input[idx];
    }
    float val = sum / D + bias[c];  // add channel bias

    // 2. Softmax over channels (dim=1)
    extern __shared__ float s_exp[];
    float exp_val = expf(val);
    s_exp[c] = exp_val;
    __syncthreads();

    // Parallel reduction of exp sum (works for any C)
    for (int s = 1; s < blockDim.x; s <<= 1) {
        if (c % (2 * s) == 0) {
            s_exp[c] += s_exp[c + s];
        }
        __syncthreads();
    }
    float sum_exp = s_exp[0];
    __syncthreads();

    float softmax_val = exp_val / sum_exp;

    // 3. Tanh and scaling
    float tanh_val = tanhf(softmax_val);
    float out_val = tanh_val * scaling_factor;

    // 4. Write to output [B, C, 1, H, W]
    int out_idx = (((b * C + c) * 1) + 0) * H * W + h * W + w; // D_out = 1
    output[out_idx] = out_val;
}

torch::Tensor fused_mean_bias_softmax_tanh_scale_cuda(
    torch::Tensor conv_out,
    torch::Tensor bias,
    float scaling_factor)
{
    int B = conv_out.size(0);
    int C = conv_out.size(1);
    int D = conv_out.size(2);
    int H = conv_out.size(3);
    int W = conv_out.size(4);

    auto output = torch::empty({B, C, 1, H, W}, conv_out.options());

    dim3 grid(H, W, B);
    dim3 block(C);
    int shared_mem_size = C * sizeof(float); // dynamic shared memory for exp values

    fused_mean_bias_softmax_tanh_scale_kernel<<<grid, block, shared_mem_size>>>(
        conv_out.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, D, H, W,
        scaling_factor
    );

    return output;
}
"""

cpp_source = "torch::Tensor fused_mean_bias_softmax_tanh_scale_cuda(torch::Tensor conv_out, torch::Tensor bias, float scaling_factor);"

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_mean_bias_softmax_tanh_scale",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_mean_bias_softmax_tanh_scale_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                 stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor
        self.fused_op = fused_op

    def forward(self, x):
        # 3D transposed convolution
        x = self.conv_transpose(x)  # shape: (B, out_channels, D, H, W)
        # Fused mean over depth, bias, softmax over channels, tanh, scaling
        x = self.fused_op.fused_mean_bias_softmax_tanh_scale_cuda(
            x, self.bias.squeeze(), self.scaling_factor
        )  # output shape: (B, out_channels, 1, H, W)
        return x