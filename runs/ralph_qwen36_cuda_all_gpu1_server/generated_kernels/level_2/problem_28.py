import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels for Linear (GEMM), InstanceNorm-like normalization, 
# and the final element-wise operations fused together.
# We will fuse: Linear -> Reshape/View -> Normalization (Mean/Var) -> Scale/Shift -> Add -> Mul
# However, since InstanceNorm2d on a 1D tensor (batch, features) effectively computes mean/var per feature 
# across the batch dimension (if we view it as N, C, H=1, W=1), the statistics are computed over the batch.
# Standard InstanceNorm normalizes each channel independently. For shape (B, C, 1, 1), it computes mean/var over B, H, W.
# Here H=W=1, so it computes mean/var over B. This is equivalent to LayerNorm if we consider the whole feature vector, 
# but InstanceNorm treats each 'channel' (feature) independently. Since H=W=1, there is only one spatial location per channel.
# So for each feature c, it computes mean and var of x[:, c] over the batch dimension.

custom_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Helper to get grid/block dimensions
dim3 get_grid_dims(int n) {
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    return dim3(num_blocks);
}

// Kernel 1: Matrix Multiplication (GEMM) for Linear Layer
// Computes out = x * w^T + b
// x: (B, I), w: (O, I), b: (O,), out: (B, O)
__global__ void linear_kernel(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b, float* __restrict__ out, int B, int I, int O) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Batch index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Output feature index

    if (row < B && col < O) {
        float sum = 0.0f;
        const float* x_row = x + row * I;
        const float* w_col = w + col * I; // w is stored as (O, I), so w[col] is the col-th row of weights
        
        // Unroll loop for performance if possible, but standard loop is fine
        for (int i = 0; i < I; ++i) {
            sum += x_row[i] * w_col[i];
        }
        
        out[row * O + col] = sum + b[col];
    }
}

// Kernel 2: Compute Mean and Variance per feature (channel) across batch
// Input: (B, C), Output: mean (C), var (C)
__global__ void compute_stats_kernel(const float* __restrict__ x, float* __restrict__ mean, float* __restrict__ var, int B, int C) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < C) {
        float sum = 0.0f;
        for (int b = 0; b < B; ++b) {
            sum += x[b * C + c];
        }
        mean[c] = sum / B;
        
        float var_sum = 0.0f;
        for (int b = 0; b < B; ++b) {
            float diff = x[b * C + c] - mean[c];
            var_sum += diff * diff;
        }
        var[c] = var_sum / B;
    }
}

// Kernel 3: Normalize, Scale, Shift, Add, Multiply
// Fused operation:
// 1. Normalize: (x - mean) / sqrt(var + eps)
// 2. Scale/Shift: gamma * normalized + beta (InstanceNorm params)
// 3. Add y: res = normed + y
// 4. Mul y: out = res * y
// Inputs: x (B, C), mean (C), var (C), gamma (C), beta (C), y (B, C)
// Output: out (B, C)
__global__ void fused_norm_add_mul_kernel(const float* __restrict__ x, const float* __restrict__ mean, const float* __restrict__ var, 
                                          const float* __restrict__ gamma, const float* __restrict__ beta, 
                                          const float* __restrict__ y, float* __restrict__ out, int B, int C, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * C) {
        int b = idx / C;
        int c = idx % C;
        
        float val = x[idx];
        float m = mean[c];
        float v = var[c];
        
        float inv_std = rsqrtf(v + eps);
        float normed = (val - m) * inv_std;
        
        float scaled = gamma[c] * normed + beta[c];
        
        float y_val = y[idx];
        float res = scaled + y_val;
        out[idx] = res * y_val;
    }
}

torch::Tensor fused_linear_norm_add_mul_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b, 
                                             torch::Tensor gamma, torch::Tensor beta, torch::Tensor y, float eps) {
    int B = x.size(0);
    int I = x.size(1);
    int O = w.size(0); // Output features
    
    TORCH_CHECK(x.size(1) == I, "Input feature mismatch");
    TORCH_CHECK(w.size(0) == O && w.size(1) == I, "Weight shape mismatch");
    TORCH_CHECK(b.size(0) == O, "Bias shape mismatch");
    TORCH_CHECK(y.size(0) == B && y.size(1) == O, "Y shape mismatch");
    TORCH_CHECK(gamma.size(0) == O && beta.size(0) == O, "Gamma/Beta shape mismatch");

    auto out = torch::empty({B, O}, x.options());
    
    // 1. Linear Layer: x @ w^T + b
    dim3 block_linear(32, 32);
    dim3 grid_linear(get_grid_dims(O), get_grid_dims(B));
    
    linear_kernel<<<grid_linear, block_linear>>>(x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), B, I, O);
    
    // Synchronize to ensure linear is done before stats computation if we were doing async, 
    // but for correctness in this simple flow, we proceed. 
    // Note: In a real production kernel, we might fuse Linear into the next step or use streams, 
    // but here we compute stats on the output of Linear.
    
    // 2. Compute Mean and Variance per feature across batch
    auto mean = torch::empty({O}, x.options());
    auto var = torch::empty({O}, x.options());
    
    int block_stats = 256;
    int grid_stats = (O + block_stats - 1) / block_stats;
    
    compute_stats_kernel<<<grid_stats, block_stats>>>(out.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), B, O);
    
    // 3. Fused Normalization, Add, Multiply
    int block_fused = 256;
    int total_elements = B * O;
    int grid_fused = (total_elements + block_fused - 1) / block_fused;
    
    fused_norm_add_mul_kernel<<<grid_fused, block_fused>>>(out.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), 
                                                           gamma.data_ptr<float>(), beta.data_ptr<float>(), y.data_ptr<float>(), out.data_ptr<float>(), B, O, eps);
                                                           
    return out;
}
"""

custom_cpp_source = "torch::Tensor fused_linear_norm_add_mul_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor gamma, torch::Tensor beta, torch::Tensor y, float eps);"

# Load the inline CUDA extension
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=custom_cpp_source,
    cuda_sources=custom_cuda_source,
    functions=["fused_linear_norm_add_mul_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"]
)

class ModelNew(nn.Module):
    """
    Optimized Model using custom fused CUDA operators.
    Replaces Linear, InstanceNorm2d (adapted for 1D), Add, and Mul with a single fused kernel pipeline.
    """
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        # Initialize weights and biases for the linear layer
        # PyTorch Linear uses Kaiming uniform initialization by default
        self.register_buffer('weight', torch.empty(out_features, in_features))
        self.register_buffer('bias', torch.empty(out_features))
        
        # InstanceNorm2d parameters: gamma (weight) and beta (bias)
        # For InstanceNorm2d with num_features=out_features, these are learned parameters.
        # We initialize them to 1 and 0 respectively as per PyTorch default.
        self.register_buffer('gamma', torch.ones(out_features))
        self.register_buffer('beta', torch.zeros(out_features))
        
        self.eps = eps
        
        # Initialize weights using Kaiming uniform like nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='linear')
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, y):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Input tensor of shape (batch_size, out_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        # Call the fused custom CUDA operator
        return fused_ops.fused_linear_norm_add_mul_cuda(x, self.weight, self.bias, self.gamma, self.beta, y, self.eps)

import math