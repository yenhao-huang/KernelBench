import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused matmul, max_pool, sum, and scaling
# Since the matmul is a large dense operation, we use cuBLAS via PyTorch for the matmul 
# and then fuse the subsequent MaxPool1d, Sum, and Scaling into a single kernel.
# This avoids multiple kernel launches and multiple global memory reads/writes.

fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void fused_post_matmul_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int out_features,
    int kernel_size,
    float scale_factor
) {
    // Each thread handles one batch element
    int b = blockIdx.x;
    if (b >= batch_size) return;

    // The input is (batch_size, out_features)
    // We need to perform MaxPool1d with kernel_size on the out_features dimension.
    // Note: MaxPool1d in PyTorch with kernel_size K on dimension L results in floor((L-K)/stride + 1) elements.
    // Default stride is kernel_size.
    
    int stride = kernel_size;
    int num_pooled_elements = (out_features - kernel_size) / stride + 1;
    
    // We need to sum these pooled elements and then scale.
    // However, the original model does:
    // x = matmul(x) -> (B, F_out)
    // x = max_pool(x.unsqueeze(1)) -> (B, 1, F_out_pooled)
    // x = sum(x, dim=1) -> (B, F_out_pooled) -- Wait, the original code says:
    // x = torch.sum(x, dim=1) where x is (B, F_out_pooled). 
    // This results in a tensor of shape (B,).
    
    float sum_val = 0.0f;
    
    for (int i = 0; i < num_pooled_elements; ++i) {
        int start_idx = i * stride;
        float max_val = -1e38f; // Very small float
        
        for (int k = 0; k < kernel_size; ++k) {
            float val = input[b * out_features + start_idx + k];
            if (val > max_val) {
                max_val = val;
            }
        }
        sum_val += max_val;
    }
    
    output[b] = sum_val * scale_factor;
}

torch::Tensor fused_post_matmul_cuda(
    torch::Tensor input, 
    int kernel_size, 
    float scale_factor
) {
    const int batch_size = input.size(0);
    const int out_features = input.size(1);
    
    auto output = torch::zeros({batch_size}, input.options());
    
    const int block_size = 1;
    const int grid_size = batch_size;
    
    fused_post_matmul_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_features,
        kernel_size,
        scale_factor
    );
    
    return output;
}
"""

fused_ops_cpp_source = "torch::Tensor fused_post_matmul_cuda(torch::Tensor input, int kernel_size, float scale_factor);"

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_post_matmul_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Optimized Model using fused CUDA kernel for post-matmul operations.
    """
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor
        self.fused_ops = fused_ops

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size,).
        """
        # Step 1: Perform the heavy matrix multiplication using highly optimized cuBLAS (via PyTorch)
        x = self.matmul(x)
        
        # Step 2: Fuse MaxPool1d, Sum, and Scaling into a single CUDA kernel
        # This reduces memory bandwidth usage significantly by reading the matmul output once.
        x = self.fused_ops.fused_post_matmul_cuda(x, self.kernel_size, self.scale_factor)
        
        return x