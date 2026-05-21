import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for fused subtract + hardswish + maxpool + mish
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ float mish(float x) {
    // mish(x) = x * tanh(softplus(x))
    float sp;
    if (x > 20.0f) {
        sp = x;
    } else {
        sp = log1pf(expf(x));
    }
    return x * tanhf(sp);
}

__global__ void fused_kernel(const float* input, float* output, float subtract_value,
                             int N, int C, int H_in, int W_in, int pool_size,
                             int H_out, int W_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * H_out * W_out;
    if (idx >= total_elements) return;

    int n = idx / (C * H_out * W_out);
    int rem = idx % (C * H_out * W_out);
    int c = rem / (H_out * W_out);
    int rem2 = rem % (H_out * W_out);
    int h_out = rem2 / W_out;
    int w_out = rem2 % W_out;

    int h_in_start = h_out * pool_size;
    int w_in_start = w_out * pool_size;

    float max_val = -INFINITY;
    for (int kh = 0; kh < pool_size; ++kh) {
        for (int kw = 0; kw < pool_size; ++kw) {
            int h_in = h_in_start + kh;
            int w_in = w_in_start + kw;
            float val = input[((n * C + c) * H_in + h_in) * W_in + w_in];
            float x = val - subtract_value;
            float x_plus_3 = x + 3.0f;
            float relu6 = fminf(fmaxf(x_plus_3, 0.0f), 6.0f);
            float hswish = x * relu6 / 6.0f;
            if (hswish > max_val) max_val = hswish;
        }
    }
    output[idx] = mish(max_val);
}

torch::Tensor fused_subtract_hardswish_maxpool_mish_cuda(
    torch::Tensor input, float subtract_value, int pool_size) {
    int N = input.size(0);
    int C = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    int H_out = H_in / pool_size;
    int W_out = W_in / pool_size;
    auto output = torch::empty({N, C, H_out, W_out}, input.options());

    int total_elements = N * C * H_out * W_out;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), subtract_value,
        N, C, H_in, W_in, pool_size, H_out, W_out);

    return output;
}
"""

fused_ops_cpp_source = "torch::Tensor fused_subtract_hardswish_maxpool_mish_cuda(torch::Tensor input, float subtract_value, int pool_size);"

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_subtract_hardswish_maxpool_mish_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernel fusing subtract, hardswish, maxpool, and mish.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = subtract_value
        self.pool_kernel_size = pool_kernel_size
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_ops.fused_subtract_hardswish_maxpool_mish_cuda(
            x, self.subtract_value, self.pool_kernel_size
        )
        return x