import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernels for Conv2d + ReLU fusion and Linear layers
# We use a fused approach to reduce memory bandwidth pressure by avoiding intermediate tensor allocations.

custom_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

// Helper for 2D grid/block indexing
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 8

// Kernel for Conv2d + ReLU Fusion
// Assumes input is NHWC or NCHW. Here we assume NCHW as per PyTorch default.
// Input: (N, C_in, H, W)
// Weight: (C_out, C_in, K, K)
// Bias: (C_out)
// Output: (N, C_out, H, W)
__global__ void conv2d_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int C_out,
    int H_in, int W_in,
    int K, int P) { // K is kernel size, P is padding
    
    int batch_idx = blockIdx.z;
    int out_y = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;
    int out_x = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;
    
    if (out_y >= H_in || out_x >= W_in) return;
    
    // Calculate input coordinates relative to padding
    int in_y_start = out_y - P;
    int in_x_start = out_x - P;
    
    const float* input_ptr = input + batch_idx * C_in * H_in * W_in;
    const float* output_ptr = output + batch_idx * C_out * H_in * W_in;
    
    for (int c_out = 0; c_out < C_out; ++c_out) {
        float sum = 0.0f;
        
        // Iterate over input channels and kernel spatial dimensions
        for (int c_in = 0; c_in < C_in; ++c_in) {
            const float* weight_ptr = weight + c_out * (C_in * K * K) + c_in * (K * K);
            
            for (int ky = 0; ky < K; ++ky) {
                int in_y = in_y_start + ky;
                if (in_y < 0 || in_y >= H_in) continue;
                
                for (int kx = 0; kx < K; ++kx) {
                    int in_x = in_x_start + kx;
                    if (in_x < 0 || in_x >= W_in) continue;
                    
                    float val = input_ptr[in_y * W_in * C_in + in_x * C_in + c_in];
                    sum += val * weight_ptr[ky * K + kx];
                }
            }
        }
        
        if (bias != nullptr) {
            sum += bias[c_out];
        }
        
        // Apply ReLU
        if (sum < 0.0f) sum = 0.0f;
        
        output_ptr[out_y * W_in * C_out + out_x * C_out + c_out] = sum;
    }
}

// Kernel for Linear Layer + ReLU Fusion
// Input: (N, D_in)
// Weight: (D_out, D_in)
// Bias: (D_out)
// Output: (N, D_out)
__global__ void linear_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int D_in, int D_out) {
    
    int batch_idx = blockIdx.y;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= N || out_idx >= D_out) return;
    
    const float* input_ptr = input + batch_idx * D_in;
    float sum = 0.0f;
    
    // Simple loop for linear combination. 
    // For large D_in, this could be optimized with shared memory or vectorized loads,
    // but for simplicity and correctness in inline code, we use a direct loop.
    // Note: In production, one would likely use cuBLAS sgemm here.
    // However, to demonstrate custom operator replacement as requested:
    for (int i = 0; i < D_in; ++i) {
        sum += input_ptr[i] * weight[out_idx * D_in + i];
    }
    
    if (bias != nullptr) {
        sum += bias[out_idx];
    }
    
    // Apply ReLU
    if (sum < 0.0f) sum = 0.0f;
    
    output[batch_idx * D_out + out_idx] = sum;
}

// Wrapper for Conv2d + ReLU
torch::Tensor conv2d_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {
    
    auto N = input.size(0);
    auto C_in = input.size(1);
    auto H_in = input.size(2);
    auto W_in = input.size(3);
    auto C_out = weight.size(0);
    auto K = weight.size(2); // Assuming square kernel
    
    auto output = torch::zeros({N, C_out, H_in, W_in}, input.options());
    
    const int block_x = 16;
    const int block_y = BLOCK_SIZE_Y;
    dim3 threads(block_x, block_y);
    dim3 blocks((W_in + block_x - 1) / block_x, (H_in + block_y - 1) / block_y, N);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    conv2d_relu_kernel<<<blocks, threads, 0, stream>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.numel() > 0 ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, C_out, H_in, W_in, K, 1 // Padding is 1 in VGG16 convs
    );
    
    return output;
}

// Wrapper for Linear + ReLU
torch::Tensor linear_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {
    
    auto N = input.size(0);
    auto D_in = input.size(1);
    auto D_out = weight.size(0);
    
    auto output = torch::zeros({N, D_out}, input.options());
    
    const int block_size = 256;
    dim3 threads(block_size);
    dim3 blocks((D_out + block_size - 1) / block_size, N);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    linear_relu_kernel<<<blocks, threads, 0, stream>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.numel() > 0 ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, D_in, D_out
    );
    
    return output;
}

"""

custom_cpp_source = """
#include <torch/extension.h>

torch::Tensor conv2d_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias);

torch::Tensor linear_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conv2d_relu", &conv2d_relu_cuda, "Conv2d + ReLU CUDA");
  m.def("linear_relu", &linear_relu_cuda, "Linear + ReLU CUDA");
}
"""

# Load the custom extension
custom_ops = load_inline(
    name="custom_vgg_ops",
    cpp_sources=custom_cpp_source,
    cuda_sources=custom_cuda_source,
    functions=["conv2d_relu", "linear_relu"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"]
)


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        Initialize the VGG16 model with custom CUDA operators.
        
        :param num_classes: The number of output classes (default is 1000 for ImageNet)
        """
        super(ModelNew, self).__init__()
        
        # We store weights and biases as parameters so they are registered correctly
        # and can be moved to GPU/CPU. The custom ops will access these tensors directly.
        
        # Helper to create conv layer params
        def make_conv_params(in_ch, out_ch):
            weight = nn.Parameter(torch.randn(out_ch, in_ch, 3, 3))
            bias = nn.Parameter(torch.zeros(out_ch))
            return weight, bias

        # Helper to create linear layer params
        def make_linear_params(in_ch, out_ch):
            weight = nn.Parameter(torch.randn(out_ch, in_ch))
            bias = nn.Parameter(torch.zeros(out_ch))
            return weight, bias

        # Block 1
        w1_1, b1_1 = make_conv_params(3, 64)
        w1_2, b1_2 = make_conv_params(64, 64)
        
        # Block 2
        w2_1, b2_1 = make_conv_params(64, 128)
        w2_2, b2_2 = make_conv_params(128, 128)
        
        # Block 3
        w3_1, b3_1 = make_conv_params(128, 256)
        w3_2, b3_2 = make_conv_params(256, 256)
        w3_3, b3_3 = make_conv_params(256, 256)
        
        # Block 4
        w4_1, b4_1 = make_conv_params(256, 512)
        w4_2, b4_2 = make_conv_params(512, 512)
        w4_3, b4_3 = make_conv_params(512, 512)
        
        # Block 5
        w5_1, b5_1 = make_conv_params(512, 512)
        w5_2, b5_2 = make_conv_params(512, 512)
        w5_3, b5_3 = make_conv_params(512, 512)
        
        # Classifier
        w6_1, b6_1 = make_linear_params(512 * 7 * 7, 4096)
        w6_2, b6_2 = make_linear_params(4096, 4096)
        w6_3, b6_3 = make_linear_params(4096, num_classes)

        # Store parameters
        self.register_parameter('w1_1', w1_1)
        self.register_parameter('b1_1', b1_1)
        self.register_parameter('w1_2', w1_2)
        self.register_parameter('b1_2', b1_2)
        
        self.register_parameter('w2_1', w2_1)
        self.register_parameter('b2_1', b2_1)
        self.register_parameter('w2_2', w2_2)
        self.register_parameter('b2_2', b2_2)
        
        self.register_parameter('w3_1', w3_1)
        self.register_parameter('b3_1', b3_1)
        self.register_parameter('w3_2', w3_2)
        self.register_parameter('b3_2', b3_2)
        self.register_parameter('w3_3', w3_3)
        self.register_parameter('b3_3', b3_3)
        
        self.register_parameter('w4_1', w4_1)
        self.register_parameter('b4_1', b4_1)
        self.register_parameter('w4_2', w4_2)
        self.register_parameter('b4_2', b4_2)
        self.register_parameter('w4_3', w4_3)
        self.register_parameter('b4_3', b4_3)
        
        self.register_parameter('w5_1', w5_1)
        self.register_parameter('b5_1', b5_1)
        self.register_parameter('w5_2', w5_2)
        self.register_parameter('b5_2', b5_2)
        self.register_parameter('w5_3', w5_3)
        self.register_parameter('b5_3', b5_3)
        
        self.register_parameter('w6_1', w6_1)
        self.register_parameter('b6_1', b6_1)
        self.register_parameter('w6_2', w6_2)
        self.register_parameter('b6_2', b6_2)
        self.register_parameter('w6_3', w6_3)
        self.register_parameter('b6_3', b6_3)

    def forward(self, x):
        """
        Forward pass of the VGG16 model using custom CUDA operators.
        
        :param x: The input tensor, shape (batch_size, 3, 224, 224)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        
        # Block 1
        x = custom_ops.conv2d_relu(x, self.w1_1, self.b1_1)
        x = custom_ops.conv2d_relu(x, self.w1_2, self.b1_2)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Block 2
        x = custom_ops.conv2d_relu(x, self.w2_1, self.b2_1)
        x = custom_ops.conv2d_relu(x, self.w2_2, self.b2_2)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Block 3
        x = custom_ops.conv2d_relu(x, self.w3_1, self.b3_1)
        x = custom_ops.conv2d_relu(x, self.w3_2, self.b3_2)
        x = custom_ops.conv2d_relu(x, self.w3_3, self.b3_3)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Block 4
        x = custom_ops.conv2d_relu(x, self.w4_1, self.b4_1)
        x = custom_ops.conv2d_relu(x, self.w4_2, self.b4_2)
        x = custom_ops.conv2d_relu(x, self.w4_3, self.b4_3)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Block 5
        x = custom_ops.conv2d_relu(x, self.w5_1, self.b5_1)
        x = custom_ops.conv2d_relu(x, self.w5_2, self.b5_2)
        x = custom_ops.conv2d_relu(x, self.w5_3, self.b5_3)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Classifier
        x = custom_ops.linear_relu(x, self.w6_1, self.b6_1)
        x = custom_ops.linear_relu(x, self.w6_2, self.b6_2)
        x = custom_ops.linear_relu(x, self.w6_3, self.b6_3)
        
        return x