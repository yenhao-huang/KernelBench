import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels
# We will fuse: ConvTranspose2d -> GlobalAvgPool -> Add Bias -> LogSumExp -> Sum -> Mul
# However, ConvTranspose2d is complex to implement from scratch efficiently in a single inline block without cuDNN.
# Instead, we will optimize the post-processing chain which is often memory-bound or has high overhead in Python loops.
# Actually, for maximum speedup on this specific small-ish batch but large spatial dimension (512x512), 
# the bottleneck is likely the memory bandwidth of the intermediate tensors.
# Let's implement a fused kernel for: GlobalAvgPool + AddBias + LogSumExp + Sum + Mul.
# Note: ConvTranspose2d will remain as PyTorch's optimized implementation (likely cuDNN) as writing a fast one from scratch is error-prone and less likely to beat cuDNN in this context without significant effort. 
# The prompt allows replacing "some" operators. We will replace the sequence of operations after conv_transpose.

fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Kernel for Global Average Pooling over H and W dimensions
// Input: (N, C, H, W) -> Output: (N, C, 1, 1)
__global__ void global_avg_pool_kernel(const float* input, float* output, int N, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * C) {
        int n = idx / C;
        int c = idx % C;
        float sum = 0.0f;
        const float* row_ptr = input + n * C * H * W + c * H * W;
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                sum += row_ptr[h * W + w];
            }
        }
        output[idx] = sum / (H * W);
    }
}

// Kernel for Add Bias, LogSumExp, Sum, and Multiply
// Input: (N, C, 1, 1) from GAP, Bias: (C, 1, 1), Output: (N,)
__global__ void post_process_kernel(const float* gap_out, const float* bias, float* final_out, int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float sum_exp = 0.0f;
        float max_val = -1e20f; // Initialize with a very small number
        
        // First pass: find max for numerical stability in LogSumExp
        const float* row_ptr = gap_out + idx * C;
        for (int c = 0; c < C; ++c) {
            float val = row_ptr[c] + bias[c];
            if (val > max_val) {
                max_val = val;
            }
        }
        
        // Second pass: compute sum(exp(val - max))
        for (int c = 0; c < C; ++c) {
            float val = row_ptr[c] + bias[c];
            sum_exp += expf(val - max_val);
        }
        
        // LogSumExp result: log(sum(exp)) + max
        float lse = logf(sum_exp) + max_val;
        
        // Sum over remaining dims (already reduced to scalar per sample in this logic if we consider the previous sum step)
        // Wait, the original code does:
        // 1. GAP -> (N, C, 1, 1)
        // 2. Add Bias -> (N, C, 1, 1)
        // 3. LogSumExp(dim=1) -> (N, 1, 1, 1)
        // 4. Sum(dim=(2,3)) -> (N, 1)
        // 5. Mul(10.0) -> (N, 1)
        
        // So for each sample N, we compute LSE over C channels.
        final_out[idx] = lse * 10.0f;
    }
}

torch::Tensor fused_post_process(torch::Tensor x, torch::Tensor bias) {
    // x shape: (N, C, H, W) from ConvTranspose2d
    auto N = x.size(0);
    auto C = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);

    // Allocate output for GAP: (N, C, 1, 1) -> flattened to (N*C)
    torch::Tensor gap_out = torch::empty({N * C}, x.options());
    
    const int block_size = 256;
    const int num_blocks_gap = (N * C + block_size - 1) / block_size;
    
    global_avg_pool_kernel<<<num_blocks_gap, block_size>>>(
        x.data_ptr<float>(), 
        gap_out.data_ptr<float>(), 
        N, C, H, W
    );
    
    // Allocate final output: (N,)
    torch::Tensor final_out = torch::empty({N}, x.options());
    
    const int num_blocks_post = (N + block_size - 1) / block_size;
    
    post_process_kernel<<<num_blocks_post, block_size>>>(
        gap_out.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        final_out.data_ptr<float>(), 
        N, C
    );
    
    // Reshape to match original output shape (N, 1)
    return final_out.view({N, 1});
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_post_process(torch::Tensor x, torch::Tensor bias);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_post_process"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"]
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs ConvTranspose2d using PyTorch's optimized backend,
    then fuses Global Average Pooling, Bias Addition, LogSumExp, Sum, and Multiplication
    into custom CUDA kernels for better performance.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # ConvTranspose2d remains as is (likely using cuDNN)
        x = self.conv_transpose(x)
        
        # Fused operation: GAP + AddBias + LogSumExp + Sum + Mul
        # Input x is (N, C, H, W)
        # Bias is (C, 1, 1)
        x = fused_ops.fused_post_process(x, self.bias)
        
        return x

def get_inputs():
    # randomly generate input tensors based on the model architecture
    batch_size = 16
    in_channels = 64
    height = width = 512
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    in_channels = 64
    out_channels = 128
    kernel_size = 3
    bias_shape = (out_channels, 1, 1)
    return [in_channels, out_channels, kernel_size, bias_shape]