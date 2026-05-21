import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for scaled mean pooling
scaled_mean_pooling_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void scaled_mean_pooling_kernel(const float* input, float* output, float multiplier, int N, int C, int H, int W) {
    // Each block handles one (n, c) pair
    int idx = blockIdx.x;
    int n = idx / C;
    int c = idx % C;
    int spatial_size = H * W;
    
    extern __shared__ float sdata[];
    
    // Each thread computes partial sum
    float sum = 0.0f;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    for (int i = tid; i < spatial_size; i += stride) {
        sum += input[((n * C + c) * H + i / W) * W + i % W];
    }
    sdata[tid] = sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        output[n * C + c] = (sdata[0] / spatial_size) * multiplier;
    }
}

torch::Tensor scaled_mean_pooling_cuda(torch::Tensor input, float multiplier) {
    // input shape: (N, C, H, W)
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    
    auto output = torch::empty({N, C, 1, 1}, input.options());
    
    const int threads = 256;
    const int blocks = N * C;
    const int shared_mem_size = threads * sizeof(float);
    
    scaled_mean_pooling_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), multiplier, N, C, H, W
    );
    
    return output;
}
"""

scaled_mean_pooling_cpp_source = (
    "torch::Tensor scaled_mean_pooling_cuda(torch::Tensor input, float multiplier);"
)

# Compile the inline CUDA code
scaled_mean_pooling = load_inline(
    name="scaled_mean_pooling",
    cpp_sources=scaled_mean_pooling_cpp_source,
    cuda_sources=scaled_mean_pooling_source,
    functions=["scaled_mean_pooling_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = multiplier
        self.scaled_mean_pooling = scaled_mean_pooling

    def forward(self, x):
        x = self.conv_transpose(x)
        # Fused operation: multiply by scalar and global average pooling (twice, but second is no-op)
        x = self.scaled_mean_pooling.scaled_mean_pooling_cuda(x, self.multiplier)
        return x