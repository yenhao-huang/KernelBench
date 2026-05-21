import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for fused max pooling (kernel=2, stride=2) + max pooling (kernel=3, stride=3) + channel sum
fused_pool_sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_pool_sum_kernel(const float* input, float* output,
                                      int N, int C, int D, int H, int W,
                                      int D_out, int H_out, int W_out,
                                      int window_size, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = N * D_out * H_out * W_out;
    if (idx >= total_output) return;

    int w = idx % W_out;
    int h = (idx / W_out) % H_out;
    int d = (idx / (W_out * H_out)) % D_out;
    int n = idx / (W_out * H_out * D_out);

    float sum = 0.0f;
    for (int c = 0; c < C; ++c) {
        float max_val = -1e30f;
        for (int di = 0; di < window_size; ++di) {
            int in_d = d * stride + di;
            if (in_d >= D) continue;
            for (int hi = 0; hi < window_size; ++hi) {
                int in_h = h * stride + hi;
                if (in_h >= H) continue;
                for (int wi = 0; wi < window_size; ++wi) {
                    int in_w = w * stride + wi;
                    if (in_w >= W) continue;
                    int in_idx = (((n * C + c) * D + in_d) * H + in_h) * W + in_w;
                    float val = input[in_idx];
                    if (val > max_val) max_val = val;
                }
            }
        }
        sum += max_val;
    }
    output[idx] = sum;
}

torch::Tensor fused_pool_sum_cuda(torch::Tensor input, int window_size, int stride) {
    // input shape: (N, C, D, H, W)
    int N = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    int D_out = (D - window_size) / stride + 1;
    int H_out = (H - window_size) / stride + 1;
    int W_out = (W - window_size) / stride + 1;

    auto output = torch::zeros({N, 1, D_out, H_out, W_out}, input.options());

    int total_output = N * D_out * H_out * W_out;
    const int block_size = 256;
    const int num_blocks = (total_output + block_size - 1) / block_size;

    fused_pool_sum_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        N, C, D, H, W, D_out, H_out, W_out, window_size, stride
    );

    return output;
}
"""

fused_pool_sum_cpp_source = "torch::Tensor fused_pool_sum_cuda(torch::Tensor input, int window_size, int stride);"

# Compile the inline CUDA code
fused_pool_sum = load_inline(
    name="fused_pool_sum",
    cpp_sources=fused_pool_sum_cpp_source,
    cuda_sources=fused_pool_sum_source,
    functions=["fused_pool_sum_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.fused_pool_sum = fused_pool_sum
        # window_size = 2*3 = 6, stride = 2*3 = 6 (product of the two pooling layers)
        self.window_size = 6
        self.stride = 6

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_pool_sum.fused_pool_sum_cuda(x, self.window_size, self.stride)
        return x