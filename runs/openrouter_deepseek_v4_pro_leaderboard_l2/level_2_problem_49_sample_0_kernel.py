import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for fused softmax + sigmoid along channel dimension
softmax_sigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void softmax_sigmoid_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int D, int H, int W)
{
    // Each block handles one spatial location (n, d, h, w)
    int spatial_size = D * H * W;
    int total_spatial = N * spatial_size;
    int idx = blockIdx.x;
    if (idx >= total_spatial) return;

    int n = idx / spatial_size;
    int rem = idx % spatial_size;
    int d = rem / (H * W);
    int rem2 = rem % (H * W);
    int h = rem2 / W;
    int w = rem2 % W;

    int spatial_stride = D * H * W;
    int base_spatial = n * C * spatial_stride + d * H * W + h * W + w;

    extern __shared__ float shared[];
    float* shared_max = shared;
    float* shared_sum = shared + blockDim.x;

    int tid = threadIdx.x;
    float local_max = -INFINITY;
    float local_sum = 0.0f;

    // First pass: find max
    for (int c = tid; c < C; c += blockDim.x) {
        float val = input[base_spatial + c * spatial_stride];
        local_max = fmaxf(local_max, val);
    }
    shared_max[tid] = local_max;
    __syncthreads();

    // Reduce max across block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }
    float global_max = shared_max[0];

    // Second pass: compute exp sum
    for (int c = tid; c < C; c += blockDim.x) {
        float val = input[base_spatial + c * spatial_stride];
        local_sum += expf(val - global_max);
    }
    shared_sum[tid] = local_sum;
    __syncthreads();

    // Reduce sum across block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    float global_sum = shared_sum[0];

    // Third pass: compute softmax + sigmoid and write output
    for (int c = tid; c < C; c += blockDim.x) {
        float val = input[base_spatial + c * spatial_stride];
        float softmax_val = expf(val - global_max) / global_sum;
        output[base_spatial + c * spatial_stride] = 1.0f / (1.0f + expf(-softmax_val));
    }
}

torch::Tensor softmax_sigmoid_cuda(torch::Tensor input) {
    // Assume input is contiguous float32 tensor of shape (N, C, D, H, W)
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");

    auto N = input.size(0);
    auto C = input.size(1);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);

    auto output = torch::empty_like(input);

    int spatial_size = D * H * W;
    int total_blocks = N * spatial_size;

    const int threads = 256;
    int shared_mem_size = 2 * threads * sizeof(float);

    softmax_sigmoid_kernel<<<total_blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D, H, W);

    return output;
}
"""

softmax_sigmoid_cpp_source = "torch::Tensor softmax_sigmoid_cuda(torch::Tensor input);"

# Compile the inline CUDA code
fused_op = load_inline(
    name="softmax_sigmoid_fusion",
    cpp_sources=softmax_sigmoid_cpp_source,
    cuda_sources=softmax_sigmoid_source,
    functions=["softmax_sigmoid_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding, bias=bias
        )
        self.fused_softmax_sigmoid = fused_op

    def forward(self, x):
        x = self.conv_transpose(x)
        # Ensure contiguous for custom kernel (conv_transpose output is usually contiguous)
        if not x.is_contiguous():
            x = x.contiguous()
        x = self.fused_softmax_sigmoid.softmax_sigmoid_cuda(x)
        return x