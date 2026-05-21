import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels for Conv3d + Softmax fusion and MaxPool3d
custom_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Helper to get index from 5D tensor (N, C, D, H, W)
#define INDEX(n, c, d, h, w, s_c, s_d, s_h, s_w) \
    (((n) * s_c + (c) * s_d + (d) * s_h + (h) * s_w) + (w))

// Kernel for Conv3d + Softmax fusion
// This kernel computes the convolution and then applies softmax along the channel dimension.
// It assumes input is (N, C_in, D, H, W) and output is (N, C_out, D', H', W')
__global__ void conv3d_softmax_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight, // Shape: (C_out, C_in, kD, kH, kW)
    const float* __restrict__ bias,  // Shape: (C_out)
    float* __restrict__ output,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out, int D_out, int H_out, int W_out,
    int kD, int kH, int kW,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w)
{
    // Each thread handles one output element (N, C_out, D_out, H_out, W_out)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C_out * D_out * H_out * W_out;

    if (idx >= total_elements) return;

    // Decode indices
    int w_out = idx % W_out;
    int temp = idx / W_out;
    int h_out = temp % H_out;
    temp = temp / H_out;
    int d_out = temp % D_out;
    int n_c = temp / D_out;
    int c_out = n_c % C_out;
    int n = n_c / C_out;

    // Calculate input spatial coordinates for the top-left of the kernel window
    int d_in_start = d_out * stride_d - pad_d;
    int h_in_start = h_out * stride_h - pad_h;
    int w_in_start = w_out * stride_w - pad_w;

    float sum_exp = 0.0f;
    float max_val = -1e20f; // Initialize with a very small number

    // First pass: Compute convolution result and find max for softmax stability
    float conv_val = 0.0f;
    
    // Unroll or loop over kernel dimensions and input channels
    for (int k_d = 0; k_d < kD; ++k_d) {
        int d_in = d_in_start + k_d;
        if (d_in < 0 || d_in >= D_in) continue;
        
        for (int k_h = 0; k_h < kH; ++k_h) {
            int h_in = h_in_start + k_h;
            if (h_in < 0 || h_in >= H_in) continue;

            for (int k_w = 0; k_w < kW; ++k_w) {
                int w_in = w_in_start + k_w;
                if (w_in < 0 || w_in >= W_in) continue;

                // Iterate over input channels
                for (int c_in = 0; c_in < C_in; ++c_in) {
                    // Get weight index: (c_out, c_in, k_d, k_h, k_w)
                    int w_idx = ((c_out * C_in + c_in) * kD + k_d) * kH * kW + k_h * kW + k_w;
                    
                    // Get input index: (n, c_in, d_in, h_in, w_in)
                    int i_idx = INDEX(n, c_in, d_in, h_in, w_in, C_in, D_in, H_in, W_in);

                    conv_val += weight[w_idx] * input[i_idx];
                }
            }
        }
    }

    // Add bias
    conv_val += bias[c_out];

    // Compute exp and sum for softmax
    float exp_val = expf(conv_val - max_val); // Using current max_val (which is effectively -inf initially, so this logic needs refinement for true parallel reduction)
    
    // Note: True online softmax requires two passes or atomic adds. 
    // For simplicity and correctness in a single kernel without shared memory reduction complexity,
    // we will compute the full convolution value first, then apply softmax.
    // However, to strictly follow "fusion" and avoid storing intermediate conv results for all channels before softmax,
    // we can just compute the final logit and apply softmax directly since softmax is element-wise across C dimension 
    // but requires normalization over C.
    
    // Since we are processing one (N, D_out, H_out, W_out) slice at a time, 
    // we cannot do softmax in one pass without knowing the sum of exps for all channels.
    // Therefore, a true 1-kernel Conv+Softmax fusion is complex because Softmax needs global reduction over C.
    // A common optimization is to fuse Conv and ReLU/Activation if it's element-wise. 
    // Softmax is NOT element-wise independent per channel; it couples channels.
    
    // Revised Strategy: 
    // 1. Implement a highly optimized Conv3d kernel that outputs logits.
    // 2. Implement a separate MaxPool3d kernel.
    // 3. Use PyTorch's native softmax or a simple custom one if needed, but the prompt asks for speedups via CUDA ops.
    // Given the constraint of "inline" and complexity, let's focus on optimizing the heavy Conv3d and the Pooling.
    // We will replace Conv3d with a custom optimized kernel and MaxPool3d with a custom kernel.
    // Softmax will be left to PyTorch or implemented simply if it becomes a bottleneck, but usually Conv is the bottleneck.
    
    output[idx] = conv_val; 
}

// Actually, let's implement a proper fused Conv3d + Softmax using two passes in one kernel launch per element? No, that's inefficient.
// Let's stick to optimizing Conv3d and MaxPool3d separately as they are the heavy hitters.
// We will write a custom Conv3d kernel and a custom MaxPool3d kernel.

__global__ void conv3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out, int D_out, int H_out, int W_out,
    int kD, int kH, int kW,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C_out * D_out * H_out * W_out;

    if (idx >= total_elements) return;

    int w_out = idx % W_out;
    int temp = idx / W_out;
    int h_out = temp % H_out;
    temp = temp / H_out;
    int d_out = temp % D_out;
    int n_c = temp / D_out;
    int c_out = n_c % C_out;
    int n = n_c / C_out;

    int d_in_start = d_out * stride_d - pad_d;
    int h_in_start = h_out * stride_h - pad_h;
    int w_in_start = w_out * stride_w - pad_w;

    float sum = 0.0f;

    for (int k_d = 0; k_d < kD; ++k_d) {
        int d_in = d_in_start + k_d;
        if (d_in < 0 || d_in >= D_in) continue;
        
        for (int k_h = 0; k_h < kH; ++k_h) {
            int h_in = h_in_start + k_h;
            if (h_in < 0 || h_in >= H_in) continue;

            for (int k_w = 0; k_w < kW; ++k_w) {
                int w_in = w_in_start + k_w;
                if (w_in < 0 || w_in >= W_in) continue;

                for (int c_in = 0; c_in < C_in; ++c_in) {
                    int w_idx = ((c_out * C_in + c_in) * kD + k_d) * kH * kW + k_h * kW + k_w;
                    int i_idx = INDEX(n, c_in, d_in, h_in, w_in, C_in, D_in, H_in, W_in);
                    sum += weight[w_idx] * input[i_idx];
                }
            }
        }
    }

    if (bias) {
        sum += bias[c_out];
    }

    output[idx] = sum;
}

__global__ void maxpool3d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out,
    int kD, int kH, int kW,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * D_out * H_out * W_out;

    if (idx >= total_elements) return;

    int w_out = idx % W_out;
    int temp = idx / W_out;
    int h_out = temp % H_out;
    temp = temp / H_out;
    int d_out = temp % D_out;
    int n_c = temp / D_out;
    int c = n_c % C;
    int n = n_c / C;

    int d_in_start = d_out * stride_d - pad_d;
    int h_in_start = h_out * stride_h - pad_h;
    int w_in_start = w_out * stride_w - pad_w;

    float max_val = -1e20f;

    for (int k_d = 0; k_d < kD; ++k_d) {
        int d_in = d_in_start + k_d;
        if (d_in < 0 || d_in >= D_in) continue;
        
        for (int k_h = 0; k_h < kH; ++k_h) {
            int h_in = h_in_start + k_h;
            if (h_in < 0 || h_in >= H_in) continue;

            for (int k_w = 0; k_w < kW; ++k_w) {
                int w_in = w_in_start + k_w;
                if (w_in < 0 || w_in >= W_in) continue;

                int i_idx = INDEX(n, c, d_in, h_in, w_in, C, D_in, H_in, W_in);
                float val = input[i_idx];
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
    }

    output[idx] = max_val;
}

// Python bindings
torch::Tensor conv3d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto N = input.size(0);
    auto C_in = input.size(1);
    auto D_in = input.size(2);
    auto H_in = input.size(3);
    auto W_in = input.size(4);

    auto C_out = weight.size(0);
    auto kD = weight.size(2);
    auto kH = weight.size(3);
    auto kW = weight.size(4);

    // Assuming stride=1, pad=0 for simplicity as per default nn.Conv3d behavior if not specified otherwise in the prompt's init?
    // The prompt uses nn.Conv3d(in_channels, out_channels, kernel_size). Default stride is 1, padding is 0.
    int stride_d = 1;
    int stride_h = 1;
    int stride_w = 1;
    int pad_d = 0;
    int pad_h = 0;
    int pad_w = 0;

    auto D_out = (D_in + 2 * pad_d - kD) / stride_d + 1;
    auto H_out = (H_in + 2 * pad_h - kH) / stride_h + 1;
    auto W_out = (W_in + 2 * pad_w - kW) / stride_w + 1;

    auto output = torch::zeros({N, C_out, D_out, H_out, W_out}, input.options());

    const int block_size = 256;
    int total_elements = N * C_out * D_out * H_out * W_out;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    conv3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, D_in, H_in, W_in,
        C_out, D_out, H_out, W_out,
        kD, kH, kW,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w
    );

    return output;
}

torch::Tensor maxpool3d_cuda(torch::Tensor input, int kernel_size) {
    auto N = input.size(0);
    auto C = input.size(1);
    auto D_in = input.size(2);
    auto H_in = input.size(3);
    auto W_in = input.size(4);

    int kD = kernel_size;
    int kH = kernel_size;
    int kW = kernel_size;
    
    // Default stride is equal to kernel size for MaxPool3d in PyTorch if not specified? 
    // Actually, default stride is None which defaults to kernel_size.
    int stride_d = kernel_size;
    int stride_h = kernel_size;
    int stride_w = kernel_size;
    
    int pad_d = 0;
    int pad_h = 0;
    int pad_w = 0;

    auto D_out = (D_in + 2 * pad_d - kD) / stride_d + 1;
    auto H_out = (H_in + 2 * pad_h - kH) / stride_h + 1;
    auto W_out = (W_in + 2 * pad_w - kW) / stride_w + 1;

    auto output = torch::zeros({N, C, D_out, H_out, W_out}, input.options());

    const int block_size = 256;
    int total_elements = N * C * D_out * H_out * W_out;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    maxpool3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D_in, H_in, W_in,
        D_out, H_out, W_out,
        kD, kH, kW,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w
    );

    return output;
}
"""

custom_cpp_source = (
    "torch::Tensor conv3d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"
    "torch::Tensor maxpool3d_cuda(torch::Tensor input, int kernel_size);"
);

# Load the inline CUDA extension
cuda_module = load_inline(
    name="custom_ops",
    cpp_sources=custom_cpp_source,
    cuda_sources=custom_cuda_source,
    functions=["conv3d_cuda", "maxpool3d_cuda"],
    verbose=False,
    extra_cflags=["-O2"],
    extra_ldflags=["-lcudart"]
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size
        
        # Initialize weights and biases manually to match nn.Conv3d behavior
        # nn.Conv3d uses Kaiming uniform initialization by default
        weight = torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size)
        nn.init.kaiming_uniform_(weight, a=0, mode='fan_in', nonlinearity='relu')
        self.register_buffer('conv_weight', weight)
        
        fan_in = in_channels * kernel_size * kernel_size * kernel_size
        bound = 1 / (fan_in ** 0.5)
        bias = torch.empty(out_channels)
        nn.init.uniform_(bias, -bound, bound)
        self.register_buffer('conv_bias', bias)

    def forward(self, x):
        # Perform custom Conv3d
        x = cuda_module.conv3d_cuda(x, self.conv_weight, self.conv_bias)
        
        # Apply Softmax (using PyTorch native as it's highly optimized and fusion with softmax is complex in a single kernel without shared memory reductions)
        x = torch.softmax(x, dim=1)
        
        # Perform custom MaxPool3d
        x = cuda_module.maxpool3d_cuda(x, self.pool_kernel_size)
        
        # Second MaxPool3d
        x = cuda_module.maxpool3d_cuda(x, self.pool_kernel_size)
        
        return x

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]