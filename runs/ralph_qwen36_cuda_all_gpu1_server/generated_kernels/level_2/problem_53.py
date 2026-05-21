import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM + Scaling + HardTanh + GELU
# This kernel performs: out = gelu(hardtanh(matmul(x, W^T) * scale))
# We assume x is (N, K), W is (M, K), so matmul is (N, M).
# To optimize for FP32, we use float.

fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Helper function for GELU approximation used in PyTorch's nn.GELU()
// The standard implementation uses: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
__device__ inline float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
}

// Helper function for HardTanh
__device__ inline float hardtanh(float x, float min_val, float max_val) {
    if (x < min_val) return min_val;
    if (x > max_val) return max_val;
    return x;
}

__global__ void fused_gemm_scaling_hardtanh_gelu_kernel(
    const float* __restrict__ x,      // Input: [batch_size, in_features]
    const float* __restrict__ weight, // Weight: [out_features, in_features] (transposed for row-major access if needed, but usually stored as [out, in])
    float* __restrict__ out,          // Output: [batch_size, out_features]
    int batch_size,
    int in_features,
    int out_features,
    float scaling_factor,
    float hardtanh_min,
    float hardtanh_max
) {
    // Each thread computes one element of the output matrix
    int row = blockIdx.y * blockDim.y + threadIdx.y; // batch index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // feature index

    if (row < batch_size && col < out_features) {
        float sum = 0.0f;
        
        // Perform the dot product for this specific output element
        // x[row, :] dot weight[col, :]
        // Note: PyTorch Linear uses weight of shape [out_features, in_features]
        // and computes input @ weight.T. So weight[col, k] is correct.
        for (int k = 0; k < in_features; ++k) {
            sum += x[row * in_features + k] * weight[col * in_features + k];
        }

        // Apply scaling
        sum *= scaling_factor;

        // Apply HardTanh
        sum = hardtanh(sum, hardtanh_min, hardtanh_max);

        // Apply GELU
        out[row * out_features + col] = gelu(sum);
    }
}

torch::Tensor fused_gemm_scaling_hardtanh_gelu_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    float scaling_factor,
    float hardtanh_min,
    float hardtanh_max
) {
    auto batch_size = x.size(0);
    auto in_features = x.size(1);
    auto out_features = weight.size(0);

    auto out = torch::zeros({batch_size, out_features}, x.options());

    const int block_size_x = 32;
    const int block_size_y = 8; // 32 * 8 = 256 threads per block
    
    dim3 block(block_size_x, block_size_y);
    dim3 grid((out_features + block_size_x - 1) / block_size_x, 
              (batch_size + block_size_y - 1) / block_size_y);

    fused_gemm_scaling_hardtanh_gelu_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        scaling_factor,
        hardtanh_min,
        hardtanh_max
    );

    return out;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_gemm_scaling_hardtanh_gelu_cuda("
    "torch::Tensor x,"
    "torch::Tensor weight,"
    "float scaling_factor,"
    "float hardtanh_min,"
    "float hardtanh_max"
    ");"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_gemm_scaling_hardtanh_gelu_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a fused GEMM, scaling, hardtanh, and GELU activation.
    """
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        # We still need to store the weight parameters so they can be registered 
        # and moved to GPU/CPU correctly. The custom kernel will access them directly.
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.scaling_factor = scaling_factor
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

    def forward(self, x):
        # Call the custom fused kernel
        return fused_ops.fused_gemm_scaling_hardtanh_gelu_cuda(
            x, 
            self.weight, 
            self.scaling_factor, 
            self.hardtanh_min, 
            self.hardtanh_max
        )