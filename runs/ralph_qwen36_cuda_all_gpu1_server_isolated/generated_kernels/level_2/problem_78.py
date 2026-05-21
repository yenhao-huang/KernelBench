import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels for ConvTranspose3d, MaxPool3d, and Sum reduction.
# We will fuse the operations where possible or implement them efficiently.
# Given the complexity of implementing a full generic ConvTranspose3d from scratch in inline CUDA 
# without external libraries like CUTLASS (which isn't available via load_inline easily), 
# we will focus on optimizing the MaxPool and Sum layers, and potentially replacing the 
# standard PyTorch ops with highly optimized custom kernels if feasible.
# However, writing a correct, generic, and fast ConvTranspose3d from scratch in a single inline block 
# is extremely error-prone and verbose. 
# Instead, we will optimize the MaxPool3d layers and the Sum operation, which are often bottlenecks 
# or can be fused. We will also implement a custom kernel for the entire sequence if possible, 
# but given the constraints, let's optimize the MaxPool and Sum.
# Actually, to demonstrate significant speedup and "custom CUDA operator" usage effectively, 
# we can replace the MaxPool3d and Sum with a single fused kernel that performs pooling and reduction.
# Or, we can replace the MaxPool3d with a custom optimized version.

# Let's implement a custom MaxPool3d kernel and a custom Sum kernel.
# To make it more impactful, let's try to fuse the two max pools and the sum into one kernel 
# if the dimensions allow, or just optimize each step.
# Given the specific dimensions (32x32x32), we can write efficient block-based kernels.

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define MAX_POOL_BLOCK_SIZE 16
#define SUM_BLOCK_SIZE 256

// Helper for max pooling
__device__ float warpReduceMax(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ float blockReduceMax(float val) {
    static __shared__ float shared[MAX_POOL_BLOCK_SIZE * MAX_POOL_BLOCK_SIZE];
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    shared[tid] = val;
    __syncthreads();

    for (int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }

    return tid == 0 ? shared[0] : -INFINITY;
}

// Custom MaxPool3d Kernel
__global__ void max_pool_3d_kernel(
    const float* input, 
    float* output, 
    int batch_size, 
    int channels, 
    int depth_in, int height_in, int width_in,
    int kernel_size, 
    int stride, 
    int padding,
    int depth_out, int height_out, int width_out) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * depth_out * height_out * width_out) return;

    // Decompose index into spatial coordinates
    int w_idx = idx % width_out;
    int h_idx = (idx / width_out) % height_out;
    int d_idx = (idx / (width_out * height_out)) % depth_out;
    int c_idx = (idx / (width_out * height_out * depth_out)) % channels;
    int b_idx = idx / (width_out * height_out * depth_out * channels);

    // Calculate input coordinates
    int w_start = w_idx * stride - padding;
    int h_start = h_idx * stride - padding;
    int d_start = d_idx * stride - padding;

    float max_val = -INFINITY;

    for (int k = 0; k < kernel_size; ++k) {
        for (int j = 0; j < kernel_size; ++j) {
            for (int i = 0; i < kernel_size; ++i) {
                int w_in = w_start + i;
                int h_in = h_start + j;
                int d_in = d_start + k;

                if (w_in >= 0 && w_in < width_in && 
                    h_in >= 0 && h_in < height_in && 
                    d_in >= 0 && d_in < depth_in) {
                    
                    int input_idx = b_idx * (channels * depth_in * height_in * width_in) +
                                    c_idx * (depth_in * height_in * width_in) +
                                    d_in * (height_in * width_in) +
                                    h_in * width_in + w_in;
                    
                    float val = input[input_idx];
                    if (val > max_val) {
                        max_val = val;
                    }
                }
            }
        }
    }

    int output_idx = idx;
    output[output_idx] = max_val;
}

// Custom Sum Kernel along dim 1 (channels)
__global__ void sum_channels_kernel(
    const float* input, 
    float* output, 
    int batch_size, 
    int channels_in, 
    int depth, int height, int width) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * depth * height * width) return;

    float sum_val = 0.0f;
    for (int c = 0; c < channels_in; ++c) {
        int input_idx = idx + c * (depth * height * width);
        sum_val += input[input_idx];
    }
    
    output[idx] = sum_val;
}

// Fused MaxPool3d and Sum Kernel for optimization
__global__ void fused_pool_sum_kernel(
    const float* input, 
    float* output, 
    int batch_size, 
    int channels_in, 
    int depth_in, int height_in, int width_in,
    int kernel_size1, int stride1, int padding1,
    int kernel_size2, int stride2, int padding2,
    int depth_out1, int height_out1, int width_out1,
    int depth_out2, int height_out2, int width_out2) {
    
    // Each thread handles one output element of the final sum tensor
    // Output shape: [batch_size, 1, depth_out2, height_out2, width_out2]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * depth_out2 * height_out2 * width_out2) return;

    int w_idx = idx % width_out2;
    int h_idx = (idx / width_out2) % height_out2;
    int d_idx = (idx / (width_out2 * height_out2)) % depth_out2;
    int b_idx = idx / (width_out2 * height_out2 * depth_out2);

    // First MaxPool3d
    int w_start1 = w_idx * stride1 - padding1;
    int h_start1 = h_idx * stride1 - padding1;
    int d_start1 = d_idx * stride1 - padding1;

    float max_val1 = -INFINITY;

    for (int k = 0; k < kernel_size1; ++k) {
        for (int j = 0; j < kernel_size1; ++j) {
            for (int i = 0; i < kernel_size1; ++i) {
                int w_in1 = w_start1 + i;
                int h_in1 = h_start1 + j;
                int d_in1 = d_start1 + k;

                if (w_in1 >= 0 && w_in1 < width_in && 
                    h_in1 >= 0 && h_in1 < height_in && 
                    d_in1 >= 0 && d_in1 < depth_in) {
                    
                    // We need to iterate over all channels for the first pool? 
                    // No, MaxPool3d is per channel. But then we sum over channels.
                    // So we need to find the max in each channel, then sum those maxes.
                    // This kernel structure is tricky because it's not a simple reduction.
                    // Let's revert to separate kernels for clarity and correctness, 
                    // or implement a more complex fused version.
                }
            }
        }
    }
}

// Since fusing ConvTranspose + Pool + Sum is very complex in inline CUDA without external libs,
// we will replace the MaxPool3d layers with custom optimized kernels and the Sum with a custom kernel.
// We will also provide a wrapper for ConvTranspose3d if needed, but PyTorch's is usually well-optimized.
// Let's stick to replacing MaxPool and Sum.

torch::Tensor max_pool_3d_cuda(
    torch::Tensor input, 
    int kernel_size, 
    int stride, 
    int padding) {
    
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto depth_in = input.size(2);
    auto height_in = input.size(3);
    auto width_in = input.size(4);

    // Calculate output dimensions
    int depth_out = (depth_in + 2 * padding - kernel_size) / stride + 1;
    int height_out = (height_in + 2 * padding - kernel_size) / stride + 1;
    int width_out = (width_in + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::zeros({batch_size, channels, depth_out, height_out, width_out}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * channels * depth_out * height_out * width_out + block_size - 1) / block_size;

    max_pool_3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        batch_size, 
        channels, 
        depth_in, height_in, width_in,
        kernel_size, 
        stride, 
        padding,
        depth_out, height_out, width_out
    );

    return output;
}

torch::Tensor sum_channels_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);

    // Output shape: [batch_size, 1, depth, height, width]
    auto output = torch::zeros({batch_size, 1, depth, height, width}, input.options());

    const int block_size = SUM_BLOCK_SIZE;
    const int num_blocks = (batch_size * depth * height * width + block_size - 1) / block_size;

    sum_channels_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        batch_size, 
        channels, 
        depth, height, width
    );

    return output;
}
"""

cuda_cpp_source = """
torch::Tensor max_pool_3d_cuda(torch::Tensor input, int kernel_size, int stride, int padding);
torch::Tensor sum_channels_cuda(torch::Tensor input);
"""

# Load the inline CUDA extension
custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cuda_cpp_source,
    cuda_sources=cuda_source,
    functions=["max_pool_3d_cuda", "sum_channels_cuda"],
    verbose=True,
    extra_cflags=["-O2"],
    extra_ldflags=["-lcudart"]
)


class ModelNew(nn.Module):
    """
    Optimized Model using custom CUDA operators for MaxPool3d and Sum.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # We keep the standard MaxPool3d for the first one if we want to test, 
        # but let's use custom for both to show optimization.
        self.max_pool1_kernel_size = 2
        self.max_pool1_stride = 2
        self.max_pool1_padding = 0
        
        self.max_pool2_kernel_size = 3
        self.max_pool2_stride = 2
        self.max_pool2_padding = 0

    def forward(self, x):
        x = self.conv_transpose(x)
        
        # Custom MaxPool3d for first pooling layer
        x = custom_ops.max_pool_3d_cuda(x, self.max_pool1_kernel_size, self.max_pool1_stride, self.max_pool1_padding)
        
        # Custom MaxPool3d for second pooling layer
        x = custom_ops.max_pool_3d_cuda(x, self.max_pool2_kernel_size, self.max_pool2_stride, self.max_pool2_padding)
        
        # Custom Sum along channel dimension (dim=1)
        x = custom_ops.sum_channels_cuda(x)
        
        return x

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]