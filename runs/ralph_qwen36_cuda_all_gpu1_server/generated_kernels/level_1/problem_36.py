import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for RMS Normalization
rms_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void rms_norm_kernel(const float* x, float* out, float* rms_buffer, int batch_size, int num_features, int seq_len, float eps) {
    // Each block handles one sample in the batch
    int idx = blockIdx.x;
    if (idx >= batch_size) return;

    const float* x_ptr = x + idx * num_features * seq_len;
    float* out_ptr = out + idx * num_features * seq_len;
    
    // Calculate RMS for this sample
    float sum_sq = 0.0f;
    int total_elements = num_features * seq_len;
    
    // Parallel reduction within the block to calculate sum of squares
    // We use shared memory for efficiency if possible, but for simplicity and correctness 
    // with arbitrary sizes, we can do a simple loop or use atomic adds if needed.
    // However, since each thread block handles one sample, we can just have threads iterate.
    
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    // First pass: calculate sum of squares using parallel reduction logic within the block
    // To keep it simple and robust for any size, we'll use a standard loop with atomic adds 
    // or just let each thread compute a partial sum and then reduce.
    // Given the constraints and typical sizes, a simple loop per thread is often fast enough 
    // if we distribute the work. But for true optimization, we should do reduction.
    
    // Let's use a shared memory approach for reduction within the block for the sum of squares.
    extern __shared__ float sdata[];
    
    // Each thread computes partial sum
    float local_sum = 0.0f;
    for (int i = tid; i < total_elements; i += stride) {
        float val = x_ptr[i];
        local_sum += val * val;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Thread 0 has the total sum of squares for this sample
    float rms = sqrtf(sdata[0] / total_elements + eps);
    
    // Second pass: normalize
    if (tid < total_elements) {
        out_ptr[tid] = x_ptr[tid] / rms;
    }
}

torch::Tensor rms_norm_cuda(torch::Tensor x, float eps) {
    auto batch_size = x.size(0);
    auto num_features = x.size(1);
    auto seq_len = x.numel() / (batch_size * num_features);
    
    auto out = torch::empty_like(x);
    
    const int block_size = 256;
    // Shared memory size: block_size floats for reduction
    const int shared_mem_size = block_size * sizeof(float);
    
    rms_norm_kernel<<<batch_size, block_size, shared_mem_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        nullptr, // rms_buffer not strictly needed if we compute per sample in kernel
        batch_size,
        num_features,
        seq_len,
        eps
    );
    
    return out;
}
"""

rms_norm_cpp_source = (
    "torch::Tensor rms_norm_cuda(torch::Tensor x, float eps);"
)

# Compile the inline CUDA code for RMS Normalization
rms_norm = load_inline(
    name="rms_norm",
    cpp_sources=rms_norm_cpp_source,
    cuda_sources=rms_norm_source,
    functions=["rms_norm_cuda"],
    verbose=False,
    extra_cflags=["-O2"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Simple model that performs RMS Normalization using custom CUDA operator.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        """
        Initializes the RMSNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            eps (float, optional): A small value added to the denominator to avoid division by zero. Defaults to 1e-5.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS Normalization to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        return rms_norm.rms_norm_cuda(x, self.eps)