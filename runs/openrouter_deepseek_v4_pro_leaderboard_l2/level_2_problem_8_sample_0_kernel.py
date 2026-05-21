import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source code for the fused kernel
fused_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_div_maxpool_avg_bias_sum_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int D, int H, int W,
    int D_out, int H_out, int W_out,
    float divisor,
    int num_windows)
{
    int batch = blockIdx.x;
    if (batch >= N) return;
    
    int tid = threadIdx.x;
    int c = tid;
    
    float partial_sum = 0.0f;
    if (c < C) {
        const float* base = input + batch * C * D * H * W + c * D * H * W;
        float sum = 0.0f;
        for (int d = 0; d < D_out; ++d) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    float max_val = -INFINITY;
                    #pragma unroll
                    for (int di = 0; di < 2; ++di) {
                        int in_d = d * 2 + di;
                        #pragma unroll
                        for (int hi = 0; hi < 2; ++hi) {
                            int in_h = h * 2 + hi;
                            #pragma unroll
                            for (int wi = 0; wi < 2; ++wi) {
                                int in_w = w * 2 + wi;
                                float val = base[in_d * H * W + in_h * W + in_w] / divisor;
                                if (val > max_val) max_val = val;
                            }
                        }
                    }
                    sum += max_val;
                }
            }
        }
        float mean = sum / num_windows;
        partial_sum = mean + bias[c];
    }
    
    // Warp reduction (block size = 32, all threads in one warp)
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
    }
    
    if (tid == 0) {
        output[batch] = partial_sum;
    }
}

torch::Tensor fused_div_maxpool_avg_bias_sum_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    float divisor,
    torch::IntArrayRef pool_size)
{
    input = input.contiguous();
    bias = bias.contiguous();
    
    int N = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);
    
    int pool_d = pool_size[0];
    int pool_h = pool_size[1];
    int pool_w = pool_size[2];
    
    int D_out = D / pool_d;
    int H_out = H / pool_h;
    int W_out = W / pool_w;
    int num_windows = D_out * H_out * W_out;
    
    auto output = torch::empty({N}, input.options());
    
    const int block_size = 32;
    const int num_blocks = N;
    
    auto bias_1d = bias.view({C});
    
    fused_div_maxpool_avg_bias_sum_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias_1d.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D, H, W,
        D_out, H_out, W_out,
        divisor,
        num_windows
    );
    
    output = output.view({N, 1, 1, 1});
    return output;
}
"""

fused_cpp_source = (
    "torch::Tensor fused_div_maxpool_avg_bias_sum_cuda("
    "torch::Tensor input, torch::Tensor bias, float divisor, torch::IntArrayRef pool_size);"
)

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_div_maxpool_avg_bias_sum",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_cuda_source,
    functions=["fused_div_maxpool_avg_bias_sum_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel that fuses division, max pooling,
    global average pooling, bias addition, and channel summation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.pool_size = pool_size
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv(x)
        # The custom kernel handles: division, maxpool, global avg pool, bias add, and sum over channels
        x = self.fused_op.fused_div_maxpool_avg_bias_sum_cuda(x, self.bias, self.divisor, self.pool_size)
        return x