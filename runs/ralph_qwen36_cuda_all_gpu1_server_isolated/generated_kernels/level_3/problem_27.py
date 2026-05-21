import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernels for Conv2d + BatchNorm2d + ReLU fusion
# This kernel performs: out = ReLU(BN(Conv(x)))
# It assumes the input is NCHW format.
conv_bn_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Helper to get channel index
__device__ inline int get_channel_idx(int n, int c, int h, int w, int C, int H, int W) {
    return ((n * C + c) * H + h) * W + w;
}

__global__ void conv_bn_relu_kernel(
    const float* input, 
    const float* weight, 
    const float* bias, 
    const float* bn_mean, 
    const float* bn_var, 
    const float* bn_weight, 
    const float* bn_bias, 
    float* output, 
    int N, int C_in, int H_in, int W_in, 
    int C_out, int K_h, int K_w, 
    int P_h, int P_w, int S_h, int S_w,
    int H_out, int W_out) {
    
    // Each thread handles one output element (N, C_out, H_out, W_out)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C_out * H_out * W_out;
    
    if (idx >= total_elements) return;
    
    // Decode indices
    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c_out = (idx / (W_out * H_out)) % C_out;
    int n = idx / (W_out * H_out * C_out);
    
    // Calculate input region bounds
    int h_start = h_out * S_h - P_h;
    int w_start = w_out * S_w - P_w;
    
    float sum = 0.0f;
    
    // Perform convolution for this specific output pixel and channel
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kh = 0; kh < K_h; ++kh) {
            int h_in = h_start + kh;
            if (h_in < 0 || h_in >= H_in) continue;
            
            for (int kw = 0; kw < K_w; ++kw) {
                int w_in = w_start + kw;
                if (w_in < 0 || w_in >= W_in) continue;
                
                // Get weight index: [C_out, C_in, K_h, K_w]
                int w_idx = ((c_out * C_in + c_in) * K_h + kh) * K_w + kw;
                float w_val = weight[w_idx];
                
                // Get input index: [N, C_in, H_in, W_in]
                int i_idx = get_channel_idx(n, c_in, h_in, w_in, C_in, H_in, W_in);
                float x_val = input[i_idx];
                
                sum += w_val * x_val;
            }
        }
    }
    
    // Add bias from Conv layer (usually added before BN in standard implementations, 
    // but here we treat 'bias' as the conv bias. If using fused conv+bn, usually conv has no bias or it's absorbed.
    // Standard PyTorch Conv2d has bias. We add it here.)
    sum += bias[c_out];
    
    // Batch Normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    float var = bn_var[c_out];
    float mean = bn_mean[c_out];
    float gamma = bn_weight[c_out];
    float beta = bn_bias[c_out];
    
    float inv_std = 1.0f / sqrtf(var + 1e-5); // eps=1e-5 is standard
    
    float normalized = (sum - mean) * inv_std;
    float result = normalized * gamma + beta;
    
    // ReLU
    if (result < 0.0f) {
        result = 0.0f;
    }
    
    output[idx] = result;
}

torch::Tensor conv_bn_relu_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    torch::Tensor bn_mean, 
    torch::Tensor bn_var, 
    torch::Tensor bn_weight, 
    torch::Tensor bn_bias) {
    
    auto N = input.size(0);
    auto C_in = input.size(1);
    auto H_in = input.size(2);
    auto W_in = input.size(3);
    
    auto C_out = weight.size(0);
    auto K_h = weight.size(2);
    auto K_w = weight.size(3);
    
    // Assuming stride=1, padding=1 for the specific architecture provided in _make_stage
    int S_h = 1;
    int S_w = 1;
    int P_h = 1;
    int P_w = 1;
    
    auto H_out = (H_in + 2 * P_h - K_h) / S_h + 1;
    auto W_out = (W_in + 2 * P_w - K_w) / S_w + 1;
    
    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());
    
    const int block_size = 256;
    int total_elements = N * C_out * H_out * W_out;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    conv_bn_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        bn_mean.data_ptr<float>(),
        bn_var.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, K_h, K_w,
        P_h, P_w, S_h, S_w,
        H_out, W_out
    );
    
    return output;
}

// MaxPool2d kernel with stride 2, kernel 2, padding 0
__global__ void max_pool_kernel(
    const float* input, 
    float* output, 
    int N, int C, int H_in, int W_in, 
    int K_h, int K_w, int S_h, int S_w, int P_h, int P_w,
    int H_out, int W_out) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * H_out * W_out;
    
    if (idx >= total_elements) return;
    
    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c = (idx / (W_out * H_out)) % C;
    int n = idx / (W_out * H_out * C);
    
    float max_val = -1e30f; // Initialize with a very small number
    
    for (int kh = 0; kh < K_h; ++kh) {
        for (int kw = 0; kw < K_w; ++kw) {
            int h_in = h_out * S_h + kh - P_h;
            int w_in = w_out * S_w + kw - P_w;
            
            if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                int i_idx = get_channel_idx(n, c, h_in, w_in, C, H_in, W_in);
                float val = input[i_idx];
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
    }
    
    output[idx] = max_val;
}

torch::Tensor max_pool_cuda(torch::Tensor input) {
    auto N = input.size(0);
    auto C = input.size(1);
    auto H_in = input.size(2);
    auto W_in = input.size(3);
    
    int K_h = 2;
    int K_w = 2;
    int S_h = 2;
    int S_w = 2;
    int P_h = 0;
    int P_w = 0;
    
    auto H_out = (H_in + 2 * P_h - K_h) / S_h + 1;
    auto W_out = (W_in + 2 * P_w - K_w) / S_w + 1;
    
    auto output = torch::zeros({N, C, H_out, W_out}, input.options());
    
    const int block_size = 256;
    int total_elements = N * C * H_out * W_out;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    max_pool_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H_in, W_in,
        K_h, K_w, S_h, S_w, P_h, P_w,
        H_out, W_out
    );
    
    return output;
}

// Global Average Pooling kernel
__global__ void gap_kernel(
    const float* input, 
    float* output, 
    int N, int C, int H_in, int W_in) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C;
    
    if (idx >= total_elements) return;
    
    int c = idx % C;
    int n = idx / C;
    
    float sum = 0.0f;
    int count = H_in * W_in;
    
    for (int h = 0; h < H_in; ++h) {
        for (int w = 0; w < W_in; ++w) {
            int i_idx = get_channel_idx(n, c, h, w, C, H_in, W_in);
            sum += input[i_idx];
        }
    }
    
    output[idx] = sum / count;
}

torch::Tensor gap_cuda(torch::Tensor input) {
    auto N = input.size(0);
    auto C = input.size(1);
    auto H_in = input.size(2);
    auto W_in = input.size(3);
    
    auto output = torch::zeros({N, C}, input.options());
    
    const int block_size = 256;
    int total_elements = N * C;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    gap_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H_in, W_in
    );
    
    return output;
}

// Linear Layer (Matmul + Bias Add)
__global__ void linear_kernel(
    const float* input, 
    const float* weight, 
    const float* bias, 
    float* output, 
    int N, int D_in, int D_out) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * D_out;
    
    if (idx >= total_elements) return;
    
    int n = idx / D_out;
    int d_out = idx % D_out;
    
    float sum = 0.0f;
    for (int d_in = 0; d_in < D_in; ++d_in) {
        int i_idx = n * D_in + d_in;
        int w_idx = d_out * D_in + d_in; // Weight is typically [D_out, D_in] in PyTorch Linear
        sum += input[i_idx] * weight[w_idx];
    }
    
    if (bias) {
        sum += bias[d_out];
    }
    
    output[idx] = sum;
}

torch::Tensor linear_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto N = input.size(0);
    auto D_in = input.size(1);
    auto D_out = weight.size(0);
    
    auto output = torch::zeros({N, D_out}, input.options());
    
    const int block_size = 256;
    int total_elements = N * D_out;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    linear_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, D_in, D_out
    );
    
    return output;
}
"""

conv_bn_relu_cpp_source = (
    "torch::Tensor conv_bn_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor bn_mean, torch::Tensor bn_var, torch::Tensor bn_weight, torch::Tensor bn_bias);"
    "torch::Tensor max_pool_cuda(torch::Tensor input);"
    "torch::Tensor gap_cuda(torch::Tensor input);"
    "torch::Tensor linear_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"
)

# Compile the inline CUDA code
custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=conv_bn_relu_cpp_source,
    cuda_sources=conv_bn_relu_source,
    functions=["conv_bn_relu_cuda", "max_pool_cuda", "gap_cuda", "linear_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    def __init__(self, input_channels, stages, block_widths, output_classes):
        """
        :param input_channels: int, Number of input channels for the first layer
        :param stages: int, Number of stages in the RegNet architecture
        :param block_widths: List[int], Width (number of channels) for each block in the stages
        :param output_classes: int, Number of output classes for classification
        """
        super(ModelNew, self).__init__()

        self.stages = stages
        self.block_widths = block_widths
        
        layers = []
        current_channels = input_channels
        
        # Construct the stages with their respective blocks
        for i in range(stages):
            layers.append(self._make_stage_custom(current_channels, block_widths[i]))
            current_channels = block_widths[i]
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Final fully connected layer parameters are stored as buffers or just used directly
        # We will handle the FC layer in forward pass using custom op
        
    def _make_stage_custom(self, in_channels, out_channels):
        """
        Creates a simple block for each stage using custom CUDA ops.
        The structure is: Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
        We fuse Conv+BN+ReLU into a single kernel for performance.
        """
        # We need to store the parameters of the layers so we can pass them to the custom ops
        # Since nn.Module handles parameter registration, we keep the modules but override forward
        
        conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        bn1 = nn.BatchNorm2d(out_channels)
        
        conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        bn2 = nn.BatchNorm2d(out_channels)
        
        # Store parameters as buffers or attributes to access them in forward
        # Note: In a real production scenario, you might want to detach these from the main graph 
        # if they are not being trained, but here we assume standard training.
        self.register_buffer('conv1_weight', conv1.weight)
        self.register_buffer('conv1_bias', conv1.bias)
        self.register_buffer('bn1_mean', bn1.running_mean)
        self.register_buffer('bn1_var', bn1.running_var)
        self.register_buffer('bn1_weight', bn1.weight)
        self.register_buffer('bn1_bias', bn1.bias)
        
        self.register_buffer('conv2_weight', conv2.weight)
        self.register_buffer('conv2_bias', conv2.bias)
        self.register_buffer('bn2_mean', bn2.running_mean)
        self.register_buffer('bn2_var', bn2.running_var)
        self.register_buffer('bn2_weight', bn2.weight)
        self.register_buffer('bn2_bias', bn2.bias)
        
        return nn.ModuleList([conv1, bn1, conv2, bn2])

    def forward(self, x):
        """
        Forward pass through the RegNet model using custom CUDA operators.
        """
        # Feature Extraction
        for i in range(0, len(self.feature_extractor), 2):
            # First Conv + BN + ReLU
            conv1 = self.feature_extractor[i]
            bn1 = self.feature_extractor[i+1]
            
            x = custom_ops.conv_bn_relu_cuda(
                x, 
                self.conv1_weight, 
                self.conv1_bias, 
                self.bn1_mean, 
                self.bn1_var, 
                self.bn1_weight, 
                self.bn1_bias
            )
            
            # Second Conv + BN + ReLU
            conv2 = self.feature_extractor[i+2]
            bn2 = self.feature_extractor[i+3]
            
            x = custom_ops.conv_bn_relu_cuda(
                x, 
                self.conv2_weight, 
                self.conv2_bias, 
                self.bn2_mean, 
                self.bn2_var, 
                self.bn2_weight, 
                self.bn2_bias
            )
            
            # MaxPool2d
            x = custom_ops.max_pool_cuda(x)
        
        # Global Average Pooling
        x = custom_ops.gap_cuda(x)
        
        # Fully Connected Layer
        # Flatten is implicit in GAP output shape (N, C)
        fc_weight = self.fc.weight
        fc_bias = self.fc.bias
        
        x = custom_ops.linear_cuda(x, fc_weight, fc_bias)
        
        return x

# Test code for the RegNet model
batch_size = 8
input_channels = 3
image_height, image_width = 224, 224
stages = 3
block_widths = [64, 128, 256]
output_classes = 10

def get_inputs():
    """ Generates random input tensor of shape (batch_size, input_channels, height, width) """
    return [torch.rand(batch_size, input_channels, image_height, image_width)]

def get_init_inputs():
    """ Initializes model parameters """
    return [input_channels, stages, block_widths, output_classes]