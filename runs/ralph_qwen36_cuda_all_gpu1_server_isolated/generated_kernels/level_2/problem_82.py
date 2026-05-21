import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation: Conv2d -> Tanh -> Scale -> Add Bias -> MaxPool2d
# Note: Standard cuDNN convolutions are highly optimized. The main opportunity here is fusing the post-processing
# steps (Tanh, Scale, Add, MaxPool) into a single kernel to avoid global memory writes/reads between them.
# We will keep the convolution as nn.Conv2d for now, but fuse everything after it.

fused_post_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Helper for tanh approximation or exact calculation
__device__ inline float fast_tanh(float x) {
    return tanhf(x);
}

// Kernel to perform: Tanh, Scale, Add Bias, and MaxPool in one pass
// Input: (N, C, H, W)
// Output: (N, C, H', W') where H' = H/pool_size, W' = W/pool_size
__global__ void fused_tanh_scale_add_pool_kernel(
    const float* input, 
    const float* bias, 
    float* output, 
    int batch_size, 
    int channels, 
    int height, 
    int width, 
    float scaling_factor, 
    int pool_kernel_size) 
{
    // Each thread handles one element of the output tensor
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int total_output_elements = batch_size * channels * (height / pool_kernel_size) * (width / pool_kernel_size);
    
    if (idx >= total_output_elements) {
        return;
    }

    // Decode indices for output tensor
    int w_out = idx % (width / pool_kernel_size);
    int h_out = (idx / (width / pool_kernel_size)) % (height / pool_kernel_size);
    int c_out = (idx / ((width / pool_kernel_size) * (height / pool_kernel_size))) % channels;
    int n_out = idx / ((width / pool_kernel_size) * (height / pool_kernel_size) * channels);

    // Calculate the starting index in the input tensor for this output element's receptive field
    int h_start = h_out * pool_kernel_size;
    int w_start = w_out * pool_kernel_size;

    float max_val = -INFINITY;

    // Iterate over the pooling window to find the maximum value after Tanh, Scale, and Bias addition
    for (int ph = 0; ph < pool_kernel_size; ++ph) {
        for (int pw = 0; pw < pool_kernel_size; ++pw) {
            int h_in = h_start + ph;
            int w_in = w_start + pw;
            
            // Index in the input tensor: N, C, H, W stride order
            int input_idx = ((n_out * channels + c_out) * height + h_in) * width + w_in;
            
            float val = input[input_idx];
            
            // Apply Tanh
            val = fast_tanh(val);
            
            // Apply Scaling
            val *= scaling_factor;
            
            // Add Bias (bias is broadcasted over H and W, so same bias for all spatial positions in a channel)
            val += bias[c_out];
            
            if (val > max_val) {
                max_val = val;
            }
        }
    }

    // Write the result to the output tensor
    int output_idx = ((n_out * channels + c_out) * (height / pool_kernel_size) + h_out) * (width / pool_kernel_size) + w_out;
    output[output_idx] = max_val;
}

torch::Tensor fused_tanh_scale_add_pool_cuda(
    torch::Tensor input, 
    torch::Tensor bias, 
    float scaling_factor, 
    int pool_kernel_size) 
{
    // Input shape: (N, C, H, W)
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D");
    
    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    // Check if pooling dimensions are valid
    TORCH_CHECK(height % pool_kernel_size == 0, "Height must be divisible by pool kernel size");
    TORCH_CHECK(width % pool_kernel_size == 0, "Width must be divisible by pool kernel size");

    int out_height = height / pool_kernel_size;
    int out_width = width / pool_kernel_size;

    auto output = torch::zeros({batch_size, channels, out_height, out_width}, input.options());

    const int block_size = 256;
    int total_output_elements = batch_size * channels * out_height * out_width;
    const int num_blocks = (total_output_elements + block_size - 1) / block_size;

    fused_tanh_scale_add_pool_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        output.data_ptr<float>(), 
        batch_size, 
        channels, 
        height, 
        width, 
        scaling_factor, 
        pool_kernel_size
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in fused_tanh_scale_add_pool: ") + cudaGetErrorString(err));
    }

    return output;
}
"""

fused_post_conv_cpp_source = (
    "torch::Tensor fused_tanh_scale_add_pool_cuda(torch::Tensor input, torch::Tensor bias, float scaling_factor, int pool_kernel_size);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_post_conv_cpp_source,
    cuda_sources=fused_post_conv_source,
    functions=["fused_tanh_scale_add_pool_cuda"],
    verbose=False,
    extra_cflags=["-O2"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    A model that performs a convolution, then fuses Tanh, scaling, bias addition, and max-pooling into a single CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        # Convolution (using standard optimized cuDNN implementation)
        x = self.conv(x)
        
        # Fused operation: Tanh -> Scale -> Add Bias -> MaxPool2d
        x = fused_ops.fused_tanh_scale_add_pool_cuda(
            x, 
            self.bias, 
            self.scaling_factor, 
            self.pool_kernel_size
        )
        
        return x


def get_inputs():
    # randomly generate input tensors based on the model architecture
    batch_size = 128
    in_channels = 8
    height, width = 256, 256
    return [torch.rand(batch_size, in_channels, height, width).cuda()]


def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    batch_size = 128
    in_channels = 8
    out_channels = 64
    height, width = 256, 256
    kernel_size = 3
    scaling_factor = 2.0
    bias_shape = (out_channels, 1, 1)
    pool_kernel_size = 4
    return [in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size]