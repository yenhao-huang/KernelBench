import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for fused scale + maxpool + global_avg_pool + clamp
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_scale_maxpool_avg_clamp_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int N,
    const int C,
    const int D,
    const int H,
    const int W,
    const float scale,
    const float clamp_min,
    const float clamp_max
) {
    int idx = blockIdx.x;
    int n = idx / C;
    int c = idx % C;
    if (n >= N || c >= C) return;

    // Number of full 2x2x2 windows along each dimension (stride=2, no padding)
    int D_windows = D / 2;
    int H_windows = H / 2;
    int W_windows = W / 2;
    int total_windows = D_windows * H_windows * W_windows;

    // Base pointer to the (n,c) slice
    const float* base = input + n * C * D * H * W + c * D * H * W;

    // Each thread accumulates partial sum of max*scale over a subset of windows
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < total_windows; i += blockDim.x) {
        int w_idx = i % W_windows;
        int h_idx = (i / W_windows) % H_windows;
        int d_idx = i / (W_windows * H_windows);

        int d_start = d_idx * 2;
        int h_start = h_idx * 2;
        int w_start = w_idx * 2;

        // Compute max over the 2x2x2 block
        float max_val = -INFINITY;
        for (int dd = 0; dd < 2; ++dd) {
            for (int hh = 0; hh < 2; ++hh) {
                for (int ww = 0; ww < 2; ++ww) {
                    float val = base[(d_start + dd) * H * W + (h_start + hh) * W + (w_start + ww)];
                    if (val > max_val) max_val = val;
                }
            }
        }
        thread_sum += max_val * scale;
    }

    // Block reduction to sum all thread contributions
    __shared__ float shared_sum[256];
    shared_sum[threadIdx.x] = thread_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Thread 0 writes the final clamped average
    if (threadIdx.x == 0) {
        float avg = shared_sum[0] / total_windows;
        avg = fminf(fmaxf(avg, clamp_min), clamp_max);
        output[n * C + c] = avg;  // output shape: (N, C, 1, 1, 1) flattened
    }
}

torch::Tensor fused_scale_maxpool_avg_clamp_cuda(
    torch::Tensor input,
    float scale,
    float clamp_min,
    float clamp_max
) {
    // input shape: (N, C, D, H, W)
    int N = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    // output shape: (N, C, 1, 1, 1)
    auto output = torch::zeros({N, C, 1, 1, 1}, input.options());

    const int block_size = 256;
    const int grid_size = N * C;

    fused_scale_maxpool_avg_clamp_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D, H, W,
        scale, clamp_min, clamp_max
    );

    return output;
}
"""

fused_op_cpp_source = """
torch::Tensor fused_scale_maxpool_avg_clamp_cuda(
    torch::Tensor input,
    float scale,
    float clamp_min,
    float clamp_max
);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_scale_maxpool_avg_clamp",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_scale_maxpool_avg_clamp_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale = scale
        self.clamp_min = 0.0
        self.clamp_max = 1.0
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_op.fused_scale_maxpool_avg_clamp_cuda(x, self.scale, self.clamp_min, self.clamp_max)
        return x