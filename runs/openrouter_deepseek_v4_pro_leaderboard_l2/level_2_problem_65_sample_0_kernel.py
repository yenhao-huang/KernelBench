import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for fused average pooling + sigmoid + sum
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_avgpool_sigmoid_sum_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int K, int H_pool, int W_pool, int total_pooled_per_batch,
    int chunk_size)
{
    extern __shared__ float sdata[];
    int batch = blockIdx.x;
    int chunk_start = blockIdx.y * blockDim.x;
    int tid = threadIdx.x;
    int idx = chunk_start + tid;
    
    float val = 0.0f;
    if (idx < total_pooled_per_batch) {
        int tmp = idx;
        int j = tmp % W_pool;
        tmp /= W_pool;
        int i = tmp % H_pool;
        tmp /= H_pool;
        int c = tmp;
        
        float sum = 0.0f;
        int in_h_start = i * K;
        int in_w_start = j * K;
        int input_hw = H * W;
        int base = batch * C * input_hw + c * input_hw;
        for (int dy = 0; dy < K; ++dy) {
            for (int dx = 0; dx < K; ++dx) {
                sum += input[base + (in_h_start + dy) * W + (in_w_start + dx)];
            }
        }
        float avg = sum / (float)(K * K);
        val = 1.0f / (1.0f + expf(-avg));
    }
    
    sdata[tid] = val;
    __syncthreads();
    
    // block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(&output[batch], sdata[0]);
    }
}

torch::Tensor fused_avgpool_sigmoid_sum_cuda(torch::Tensor input, int K) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    
    int H_pool = H / K;
    int W_pool = W / K;
    int total_pooled = C * H_pool * W_pool;
    
    auto output = torch::zeros({N}, input.options());
    
    const int blockDim = 256;
    int gridY = (total_pooled + blockDim - 1) / blockDim;
    dim3 grid(N, gridY);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    fused_avgpool_sigmoid_sum_kernel<<<grid, blockDim, blockDim * sizeof(float), stream>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        K, H_pool, W_pool, total_pooled,
        blockDim
    );
    
    return output;
}
"""

cpp_source = """
torch::Tensor fused_avgpool_sigmoid_sum_cuda(torch::Tensor input, int K);
"""

# Compile the inline CUDA code
fused_module = load_inline(
    name="fused_avgpool_sigmoid_sum",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_avgpool_sigmoid_sum_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model: convolution followed by a fused kernel that performs
    average pooling, sigmoid, and sum over spatial/channel dimensions.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.pool_kernel_size = pool_kernel_size
        self.fused_op = fused_module

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_op.fused_avgpool_sigmoid_sum_cuda(x, self.pool_kernel_size)
        return x