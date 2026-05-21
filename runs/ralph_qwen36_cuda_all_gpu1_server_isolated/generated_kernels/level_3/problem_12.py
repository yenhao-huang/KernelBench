import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernels for Conv2d + ReLU fusion and Linear layers
# We use a fused approach to reduce memory bandwidth pressure by avoiding intermediate tensor allocations.

custom_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s at line %d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Kernel for Conv2d + ReLU Fusion
// Assumes NHWC layout internally for coalesced access if possible, but here we stick to NCHW input/output 
// and process in tiles. For simplicity and correctness with standard PyTorch NCHW inputs:
// Input: (N, C_in, H_in, W_in)
// Weight: (C_out, C_in, K_h, K_w)
// Output: (N, C_out, H_out, W_out)
// Bias: (C_out)

__global__ void conv2d_relu_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int K_h, int K_w,
    int stride_h, int stride_w, int pad_h, int pad_w,
    int H_out, int W_out) 
{
    // Each thread handles one output element (N, C_out, H_out, W_out)
    // However, to optimize memory access, we often use a 2D grid where threads in a block work on a tile.
    // Given the complexity of implementing a fully optimized tiled conv from scratch in inline code without libraries like CUTLASS,
    // we will implement a straightforward but efficient version that leverages shared memory for weights if possible, 
    // or simply optimize the access pattern. 
    
    // For this specific optimization task, we focus on fusing ReLU and reducing overhead.
    // A simple 1D mapping: idx -> (n, c_out, h_out, w_out)
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C_out * H_out * W_out;
    
    if (idx >= total_elements) return;

    // Decode indices
    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c_out_idx = (idx / (W_out * H_out)) % C_out;
    int n_idx = idx / (W_out * H_out * C_out);

    float sum = 0.0f;
    
    // Iterate over input channels and kernel dimensions
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kh = 0; kh < K_h; ++kh) {
            for (int kw = 0; kw < K_w; ++kw) {
                // Calculate input coordinates
                int h_in = h_out * stride_h - pad_h + kh;
                int w_in = w_out * stride_w - pad_w + kw;
                
                // Check bounds
                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    int input_idx = ((n_idx * C_in + c_in) * H_in + h_in) * W_in + w_in;
                    int weight_idx = ((c_out_idx * C_in + c_in) * K_h + kh) * K_w + kw;
                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Add bias and apply ReLU
    if (bias != nullptr) {
        sum += bias[c_out_idx];
    }
    
    if (sum < 0.0f) sum = 0.0f;
    
    int output_idx = ((n_idx * C_out + c_out_idx) * H_out + h_out) * W_out + w_out;
    output[output_idx] = sum;
}

// Kernel for Linear Layer (Matmul) + ReLU Fusion
// Input: (N, in_features)
// Weight: (out_features, in_features)
// Bias: (out_features)
// Output: (N, out_features)

__global__ void linear_relu_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ output,
    int N, int in_features, int out_features) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * out_features;
    
    if (idx >= total_elements) return;

    int n_idx = idx / out_features;
    int o_idx = idx % out_features;

    float sum = 0.0f;
    
    // Dot product of input row and weight column
    for (int i = 0; i < in_features; ++i) {
        int input_idx = n_idx * in_features + i;
        int weight_idx = o_idx * in_features + i;
        sum += input[input_idx] * weight[weight_idx];
    }
    
    if (bias != nullptr) {
        sum += bias[o_idx];
    }
    
    if (sum < 0.0f) sum = 0.0f;
    
    output[idx] = sum;
}

// Python bindings

torch::Tensor conv2d_relu_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias,
    int stride_h, int stride_w, int pad_h, int pad_w) 
{
    auto N = input.size(0);
    auto C_in = input.size(1);
    auto H_in = input.size(2);
    auto W_in = input.size(3);
    
    auto C_out = weight.size(0);
    auto K_h = weight.size(2);
    auto K_w = weight.size(3);
    
    auto H_out = (H_in + 2 * pad_h - K_h) / stride_h + 1;
    auto W_out = (W_in + 2 * pad_w - K_w) / stride_w + 1;
    
    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());
    
    const int block_size = 256;
    int total_elements = N * C_out * H_out * W_out;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    // If bias is empty or not provided, pass nullptr
    float* bias_ptr = nullptr;
    if (bias.numel() > 0) {
        bias_ptr = bias.data_ptr<float>();
    }

    conv2d_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, K_h, K_w,
        stride_h, stride_w, pad_h, pad_w,
        H_out, W_out
    );
    
    CUDA_CHECK(cudaGetLastError());
    return output;
}

torch::Tensor linear_relu_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias) 
{
    auto N = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    
    auto output = torch::zeros({N, out_features}, input.options());
    
    const int block_size = 256;
    int total_elements = N * out_features;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    float* bias_ptr = nullptr;
    if (bias.numel() > 0) {
        bias_ptr = bias.data_ptr<float>();
    }

    linear_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, in_features, out_features
    );
    
    CUDA_CHECK(cudaGetLastError());
    return output;
}

"""

custom_cpp_source = """
torch::Tensor conv2d_relu_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias,
    int stride_h, int stride_w, int pad_h, int pad_w);

torch::Tensor linear_relu_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias);
"""

# Load the custom extensions
custom_ops = load_inline(
    name="custom_vgg_ops",
    cpp_sources=custom_cpp_source,
    cuda_sources=custom_cuda_source,
    functions=["conv2d_relu_cuda", "linear_relu_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"]
)


class Conv2dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv2dReLU, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
        # Initialize weights and biases manually to match PyTorch defaults or standard initialization
        # Using Kaiming uniform for Conv2d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *self.kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='relu')
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        return custom_ops.conv2d_relu_cuda(
            x, 
            self.weight, 
            self.bias if self.bias is not None else torch.empty(0),
            self.stride[0], self.stride[1], self.padding[0], self.padding[1]
        )


class LinearReLU(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearReLU, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='relu')
        
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            self.bias = nn.Parameter(torch.zeros(out_features))
            # Note: PyTorch Linear init for bias is usually zeros, but let's stick to standard
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        return custom_ops.linear_relu_cuda(
            x, 
            self.weight, 
            self.bias if self.bias is not None else torch.empty(0)
        )

import math

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        Initialize the VGG19 model with custom fused CUDA operators.
        """
        super(ModelNew, self).__init__()
        
        # Helper to create Conv2dReLU blocks
        def conv_block(in_ch, out_ch, count):
            layers = []
            for i in range(count):
                stride = 1 if i == 0 else 1 # VGG usually uses stride 1 for convs within block
                padding = 1
                layers.append(Conv2dReLU(in_ch if i == 0 else out_ch, out_ch, kernel_size=3, stride=stride, padding=padding))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            # Block 1: 64 channels
            conv_block(3, 64, 2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2: 128 channels
            conv_block(64, 128, 2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3: 256 channels
            conv_block(128, 256, 4),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4: 512 channels
            conv_block(256, 512, 4),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5: 512 channels
            conv_block(512, 512, 4),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            LinearReLU(512 * 7 * 7, 4096),
            # Dropout is removed as per the prompt's get_init_inputs suggesting p=0.0 in original, 
            # and custom ops don't handle dropout easily without more complex kernels. 
            # The original code had nn.Dropout(p=0.0) which effectively does nothing.
            LinearReLU(4096, 4096),
            nn.Linear(4096, num_classes) # Last layer usually doesn't have ReLU in classification heads, but let's check original.
                                        # Original: Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear.
                                        # The last linear does NOT have ReLU.
        )
        
        # Override the last layer to be a standard Linear without ReLU fusion if needed, 
        # or just use standard PyTorch Linear for the final output as it's small compared to convs.
        # To keep it simple and consistent with "replace operators", we can leave the last one standard 
        # or fuse it too. Since there's no activation after the last layer in standard VGG classification,
        # we will use a standard Linear for the final output to ensure correctness of logits.
        self.classifier[-1] = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def get_inputs():
    batch_size = 10
    return [torch.rand(batch_size, 3, 224, 224)]

def get_init_inputs():
    return [1000]