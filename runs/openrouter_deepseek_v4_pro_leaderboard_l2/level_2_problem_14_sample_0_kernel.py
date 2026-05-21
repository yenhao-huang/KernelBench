import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused matmul, divide, sum, and scale
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_matmul_div_sum_scale_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int batch_size,
    int input_size,
    int hidden_size,
    float scaling_factor
) {
    // Each block handles one row of the output (one sample in batch)
    int row = blockIdx.x;
    if (row >= batch_size) return;

    // Thread-local accumulator for the sum
    float sum = 0.0f;

    // Compute matmul for this row, divide by 2, and accumulate sum
    for (int h = 0; h < hidden_size; ++h) {
        float dot = 0.0f;
        const float* x_row = x + row * input_size;
        const float* w_row = weight + h * input_size;
        
        // Vectorized dot product
        for (int i = threadIdx.x; i < input_size; i += blockDim.x) {
            dot += x_row[i] * w_row[i];
        }
        
        // Warp reduction for dot product
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            dot += __shfl_down_sync(0xffffffff, dot, offset);
        }
        
        // First thread in warp writes the result
        if ((threadIdx.x & (warpSize - 1)) == 0) {
            // Divide by 2 and add to sum
            sum += dot / 2.0f;
        }
    }
    
    // Final reduction across warps (only needed if blockDim.x > warpSize)
    __shared__ float shared_sum[32]; // Max 32 warps per block
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    
    if (lane_id == 0) {
        shared_sum[warp_id] = sum;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        sum = (lane_id < (blockDim.x / warpSize)) ? shared_sum[lane_id] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane_id == 0) {
            out[row] = sum * scaling_factor;
        }
    }
}

torch::Tensor fused_matmul_div_sum_scale_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    float scaling_factor
) {
    int batch_size = x.size(0);
    int input_size = x.size(1);
    int hidden_size = weight.size(0);
    
    auto out = torch::empty({batch_size, 1}, x.options());
    
    const int threads = 256;
    const int blocks = batch_size;
    
    fused_matmul_div_sum_scale_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        input_size,
        hidden_size,
        scaling_factor
    );
    
    return out;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_matmul_div_sum_scale_cuda(torch::Tensor x, torch::Tensor weight, float scaling_factor);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_matmul_div_sum_scale_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor
        self.fused_ops = fused_ops

    def forward(self, x):
        return self.fused_ops.fused_matmul_div_sum_scale_cuda(x, self.weight, self.scaling_factor)