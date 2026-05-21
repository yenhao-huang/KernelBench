import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation:
# x = bn(conv(x) * tanh(sp(x)) * conv(x))
# Note: The original code is: x = self.bn(torch.multiply(torch.tanh(torch.nn.functional.softplus(x)), x))
# where x is the output of conv. Let y = conv(x). Then z = y * tanh(softplus(y)) * y = y^2 * tanh(softplus(y)).
# Then bn(z) is applied.

fused_conv_bn_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Helper for softplus: log(1 + exp(x))
__device__ inline float softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return 0.0f;
    return log1pf(expf(x));
}

// Helper for tanh
__device__ inline float my_tanh(float x) {
    // Use standard tanh or a fast approximation if needed, but std::tanh is usually fine on GPU via device function
    // However, to avoid linking issues with C++ math library in some environments, we can use the built-in __tanhf
    return __tanhf(x);
}

__global__ void fused_conv_bn_activation_kernel(
    const float* conv_out, 
    float* bn_out, 
    int batch_size, 
    int channels, 
    int height, 
    int width,
    float eps,
    float running_mean, // Placeholder for simplicity, usually BN requires stats. 
                        // Since we are replacing the whole block including BN, we need to handle BN stats.
                        // However, standard PyTorch BN uses running mean/var during eval and batch stats during train.
                        // To keep this self-contained and simple without external state management in the kernel signature,
                        // we will assume inference mode (using running mean/var) or pass them as arguments.
                        // For a robust solution, we should pass gamma, beta, running_mean, running_var.
    float gamma,
    float beta,
    float running_mean_val,
    float running_var_val
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * height * width;

    if (idx < total_elements) {
        // Calculate spatial index to find the channel and batch index for BN stats
        // Layout is NCHW. 
        // idx = n * C * H * W + c * H * W + h * W + w
        
        int n = idx / (channels * height * width);
        int rem = idx % (channels * height * width);
        int c = rem / (height * width);
        
        // For BN, the mean and var are per channel. 
        // In inference, we use running_mean[c] and running_var[c].
        float mean = running_mean_val; // Simplified: assuming single value or passed array? 
                                       // To be correct, we should pass pointers to arrays.
        
        // Let's refine the kernel signature to accept pointers for BN stats
    }
}

// Better approach: Pass pointers for BN parameters
__global__ void fused_conv_bn_activation_kernel_v2(
    const float* conv_out, 
    float* bn_out, 
    int batch_size, 
    int channels, 
    int height, 
    int width,
    float eps,
    const float* running_mean,
    const float* running_var,
    const float* gamma,
    const float* beta
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * height * width;

    if (idx < total_elements) {
        int n = idx / (channels * height * width);
        int rem = idx % (channels * height * width);
        int c = rem / (height * width);
        
        // Get BN stats for this channel
        float mean = running_mean[c];
        float var = running_var[c];
        float g = gamma[c];
        float b = beta[c];

        float y = conv_out[idx];
        
        // Compute softplus(y)
        float sp_y;
        if (y > 20.0f) {
            sp_y = y;
        } else if (y < -20.0f) {
            sp_y = 0.0f;
        } else {
            sp_y = log1pf(expf(y));
        }
        
        // Compute tanh(softplus(y))
        float tanh_sp_y = __tanhf(sp_y);
        
        // Compute y * tanh(softplus(y)) * y = y^2 * tanh(softplus(y))
        float z = y * y * tanh_sp_y;
        
        // Apply Batch Normalization: gamma * (z - mean) / sqrt(var + eps) + beta
        float inv_std = rsqrtf(var + eps);
        float normalized = g * (z - mean) * inv_std + b;
        
        bn_out[idx] = normalized;
    }
}

torch::Tensor fused_conv_bn_activation_cuda(
    torch::Tensor conv_out, 
    torch::Tensor running_mean, 
    torch::Tensor running_var, 
    torch::Tensor gamma, 
    torch::Tensor beta,
    float eps
) {
    auto size = conv_out.numel();
    auto out = torch::empty_like(conv_out);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_conv_bn_activation_kernel_v2<<<num_blocks, block_size>>>(
        conv_out.data_ptr<float>(), 
        out.data_ptr<float>(), 
        conv_out.size(0), // batch_size
        conv_out.size(1), // channels
        conv_out.size(2), // height
        conv_out.size(3), // width
        eps,
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>()
    );

    return out;
}
"""

fused_conv_bn_activation_cpp_source = (
    "torch::Tensor fused_conv_bn_activation_cuda("
    "torch::Tensor conv_out, "
    "torch::Tensor running_mean, "
    "torch::Tensor running_var, "
    "torch::Tensor gamma, "
    "torch::Tensor beta, "
    "float eps"
    ");"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_conv_bn_activation",
    cpp_sources=fused_conv_bn_activation_cpp_source,
    cuda_sources=fused_conv_bn_activation_source,
    functions=["fused_conv_bn_activation_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution, applies activation (y^2 * tanh(softplus(y))), 
    and then applies Batch Normalization using a custom fused CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        
        # We need to manually manage the BN parameters because we are fusing them into the CUDA kernel
        # and bypassing nn.BatchNorm2d's forward pass.
        self.bn_eps = eps
        
        # Initialize running mean and var as buffers so they are saved/loaded with state dict
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))
        
        # Gamma and Beta are learnable parameters
        self.gamma = nn.Parameter(torch.ones(out_channels))
        self.beta = nn.Parameter(torch.zeros(out_channels))
        
        self.momentum = momentum
        
    def forward(self, x):
        # 1. Convolution
        conv_out = self.conv(x)
        
        # 2. Fused Activation and Batch Normalization
        # Note: In training mode, standard BN uses batch statistics. 
        # The custom kernel above is designed for inference (using running stats).
        # To support training properly with a custom kernel, we would need to compute batch mean/var inside the kernel or call separate kernels.
        # Given the complexity of fusing training-mode BN (which requires reduction across spatial/batch dims) into a single simple kernel without external libraries like CUDNN,
        # and the prompt asking for optimization, we will assume this optimized path is primarily for inference or that we replace the standard behavior with an inference-optimized version.
        # However, to be strictly correct with PyTorch semantics including training, one might need a more complex kernel.
        # For the sake of this exercise and typical "speedup" scenarios in deployment, we implement the inference path which is the most common target for such fusion.
        
        if self.training:
            # Fallback to standard layers for training to ensure correctness of gradients and batch stats calculation
            # This is a pragmatic choice because writing a correct fused training BN kernel from scratch is extremely complex.
            x = torch.multiply(torch.tanh(torch.nn.functional.softplus(conv_out)), conv_out)
            x = nn.functional.batch_norm(x, self.running_mean, self.running_var, self.gamma, self.beta, self.training, self.momentum, self.bn_eps)
        else:
            # Use custom fused kernel for inference speedup
            x = fused_ops.fused_conv_bn_activation_cuda(
                conv_out, 
                self.running_mean, 
                self.running_var, 
                self.gamma, 
                self.beta,
                self.bn_eps
            )
            
        return x