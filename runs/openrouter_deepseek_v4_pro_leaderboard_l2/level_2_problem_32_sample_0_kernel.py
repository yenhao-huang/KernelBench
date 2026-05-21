import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for fused scale and channel-wise minimum
scale_and_min_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void scale_and_min_kernel(const float* __restrict__ input, float* __restrict__ output,
                                     float scale, int N, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_spatial = H * W;
    int total_elements = N * total_spatial;
    if (idx < total_elements) {
        int n = idx / total_spatial;
        int spatial_idx = idx % total_spatial;
        int h = spatial_idx / W;
        int w = spatial_idx % W;
        
        float min_val = INFINITY;
        for (int c = 0; c < C; ++c) {
            float val = input[n * C * H * W + c * H * W + h * W + w] * scale;
            if (val < min_val) min_val = val;
        }
        output[n * total_spatial + h * W + w] = min_val;
    }
}

torch::Tensor scale_and_min_cuda(torch::Tensor input, float scale) {
    auto N = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);
    
    auto output = torch::empty({N, 1, H, W}, input.options());
    
    int total_elements = N * H * W;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    scale_and_min_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), scale, N, C, H, W);
    
    return output;
}
"""

scale_and_min_cpp_source = "torch::Tensor scale_and_min_cuda(torch::Tensor input, float scale);"

# Compile the inline CUDA code
scale_and_min = load_inline(
    name="scale_and_min",
    cpp_sources=scale_and_min_cpp_source,
    cuda_sources=scale_and_min_source,
    functions=["scale_and_min_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.scale_and_min = scale_and_min

    def forward(self, x):
        x = self.conv(x)
        # Fused scale and channel-wise minimum
        x = self.scale_and_min.scale_and_min_cuda(x, self.scale_factor)
        return x