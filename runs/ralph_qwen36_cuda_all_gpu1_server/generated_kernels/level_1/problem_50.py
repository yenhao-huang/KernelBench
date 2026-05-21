import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Conv2d with stride 4, padding 2, kernel 11x11
# This is a naive but functional implementation. For production, one would use cuDNN or CUTLASS.
# Here we implement a simple im2col + gemm approach or direct convolution.
# Given the constraints and "imagination" clause, we will write a direct convolution kernel 
# that handles the specific dimensions efficiently using shared memory if possible, 
# but for simplicity and correctness in inline code, we'll use a straightforward grid-stride loop 
# optimized for the specific case (3 input channels, 96 output channels).

conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Kernel for Conv2d: Input [N, C_in, H, W], Weight [C_out, C_in, K, K], Output [N, C_out, H', W']
// Stride=4, Padding=2, Kernel=11x11
__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias, // Can be null if no bias
    float* __restrict__ output,
    int N, int C_in, int H, int W,
    int C_out, int K, int stride, int pad) {
    
    // Each thread computes one element of the output tensor
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C_out * ((H - K + 2 * pad) / stride + 1) * ((W - K + 2 * pad) / stride + 1);
    
    if (idx >= total_elements) return;

    // Decode index to N, C_out, H_out, W_out
    int w_out = idx % ((H - K + 2 * pad) / stride + 1);
    int h_out = (idx / ((H - K + 2 * pad) / stride + 1)) % ((H - K + 2 * pad) / stride + 1); // Wait, standard layout is N, C, H, W. 
    // Let's stick to row-major: idx = n * (C_out * H_out * W_out) + c_out * (H_out * W_out) + h_out * W_out + w_out
    
    int H_out = (H - K + 2 * pad) / stride + 1;
    int W_out = (W - K + 2 * pad) / stride + 1;
    
    int w_out_correct = idx % W_out;
    int h_out_correct = (idx / W_out) % H_out;
    int c_out_correct = (idx / (H_out * W_out)) % C_out;
    int n_correct = idx / (C_out * H_out * W_out);

    // Calculate input coordinates
    int h_in_start = h_out_correct * stride - pad;
    int w_in_start = w_out_correct * stride - pad;

    float sum = 0.0f;
    
    // Loop over input channels and kernel dimensions
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int k_h = 0; k_h < K; ++k_h) {
            int h_in = h_in_start + k_h;
            // Check bounds for height
            if (h_in < 0 || h_in >= H) continue;
            
            for (int k_w = 0; k_w < K; ++k_w) {
                int w_in = w_in_start + k_w;
                // Check bounds for width
                if (w_in < 0 || w_in >= W) continue;

                // Fetch weight: [c_out, c_in, k_h, k_w]
                float w = weight[c_out_correct * C_in * K * K + c_in * K * K + k_h * K + k_w];
                
                // Fetch input: [n, c_in, h_in, w_in]
                float inp = input[n_correct * C_in * H * W + c_in * H * W + h_in * W + w_in];
                
                sum += w * inp;
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[c_out_correct];
    }

    output[idx] = sum;
}

torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto N = input.size(0);
    auto C_in = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);
    
    auto C_out = weight.size(0);
    auto K = weight.size(2); // Assuming square kernel
    
    int stride = 4;
    int pad = 2;
    
    auto H_out = (H - K + 2 * pad) / stride + 1;
    auto W_out = (W - K + 2 * pad) / stride + 1;
    
    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());
    
    const int block_size = 256;
    int total_elements = N * C_out * H_out * W_out;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    float* bias_ptr = nullptr;
    if (bias.numel() > 0) {
        bias_ptr = bias.data_ptr<float>();
    }

    conv2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, C_in, H, W,
        C_out, K, stride, pad
    );
    
    cudaDeviceSynchronize();
    return output;
}
"""

conv2d_cpp_source = (
    "torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"
)

# Compile the inline CUDA code for Conv2d
conv2d_module = load_inline(
    name="conv2d_custom",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_source,
    functions=["conv2d_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        # We keep the PyTorch Conv2d layer to get the weights and bias initialized correctly.
        # However, we will replace the forward pass with our custom CUDA operator.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2, bias=True)
        
    def forward(self, x):
        # Use the custom CUDA operator instead of the standard PyTorch conv2d
        return conv2d_module.conv2d_cuda(x, self.conv1.weight, self.conv1.bias)

# Test code is not included as per instructions, but get_inputs and get_init_inputs are preserved for context if needed by runner.
def get_inputs():
    return [torch.rand(256, 3, 224, 224)]

def get_init_inputs():
    return [1000]