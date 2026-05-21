import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernels for Conv2d + BatchNorm2d + ReLU fusion
# This reduces memory traffic by avoiding intermediate tensor allocations.

custom_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

// Helper to get grid/block dimensions
dim3 get_grid_dim(int n) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    return dim3(grid_size);
}

// Kernel for Conv2d + BatchNorm2d + ReLU
// Assumes input is NHWC or NCHW. Here we assume NCHW as per PyTorch default.
// We perform: out = ReLU(BN(Conv(x)))
// BN formula: y = (x - mean) / sqrt(var + eps) * gamma + beta
// To optimize, we precompute scale = gamma / sqrt(var + eps) and shift = beta - mean * scale.
// Then y = x * scale + shift.
// Finally ReLU(y).

__global__ void conv_bn_relu_kernel(
    const float* __restrict__ input,   // NCHW
    const float* __restrict__ weight,  // OIHW
    const float* __restrict__ bias,    // Optional, usually BN handles shift, but here we fuse BN params
    const float* __restrict__ bn_mean,
    const float* __restrict__ bn_var,
    const float* __restrict__ bn_gamma,
    const float* __restrict__ bn_beta,
    float* __restrict__ output,        // NCHW
    int N, int C, int H, int W,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    float eps
) {
    // Each thread handles one output pixel (n, c_out, h_out, w_out)
    // However, convolution requires reading a patch. 
    // A more efficient approach for small kernels is to have threads handle output pixels and load patches into shared memory,
    // or simply use a naive global memory access pattern if optimization isn't extreme.
    // Given the complexity of writing a fully optimized tiled conv in inline CUDA within this context,
    // we will implement a simplified but correct version that fuses BN and ReLU after a standard PyTorch Conv2d call?
    // No, the prompt asks to replace operators. 
    // Writing a full tiled 2D convolution from scratch is error-prone and lengthy.
    // Instead, let's focus on replacing the BatchNorm + ReLU + Add (Residual) pattern which is very common in ResNet.
    // Or, we can use torch.nn.functional.conv2d for the heavy lifting but fuse BN+ReLU+Add?
    // Actually, the most significant speedup in ResNet often comes from fusing Conv+BN+ReLU.
    
    // Let's implement a custom kernel that performs:
    // 1. Element-wise operations for BN (scale and shift)
    // 2. ReLU
    // 3. Addition with identity
    
    // But wait, the convolution itself is the bottleneck. 
    // Replacing Conv2d with a custom CUDA kernel is complex.
    // Let's replace the residual addition + final ReLU with a fused kernel, as it's simpler and still beneficial.
    // Also, we can fuse BN+ReLU into a single pass if we had the pre-BN tensor.
    
    // Strategy: 
    // 1. Keep Conv2d as is (cuDNN is highly optimized).
    // 2. Replace the sequence: bn -> relu -> add_identity -> relu with a custom kernel.
    // This avoids multiple memory writes/reads for intermediate tensors.
}

// Kernel: out = ReLU( BN(Conv(x)) + Identity )
// We assume the input to this kernel is the output of Conv2d (before BN) and the identity tensor.
// Wait, BN needs mean/var/gamma/beta. 
// Let's define a kernel that takes:
// - conv_out: Output of Conv2d (NCHW)
// - bn_mean, bn_var, bn_gamma, bn_beta: BatchNorm parameters
// - identity: The residual connection input (NCHW)
// - output: Result

__global__ void fused_bn_relu_add_kernel(
    const float* __restrict__ conv_out,
    const float* __restrict__ bn_mean,
    const float* __restrict__ bn_var,
    const float* __restrict__ bn_gamma,
    const float* __restrict__ bn_beta,
    const float* __restrict__ identity,
    float* __restrict__ output,
    int N, int C, int H, int W,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * H * W;
    
    if (idx < total_elements) {
        // Calculate channel index for BN parameters
        // The linear index idx corresponds to a specific pixel in the batch.
        // We need to map idx to (n, c, h, w) to get the correct BN params for channel c.
        
        int temp = idx;
        int w = temp % W;
        temp /= W;
        int h = temp % H;
        temp /= H;
        int c = temp % C;
        int n = temp / C; // Not strictly needed for BN params if we assume per-channel
        
        // Get BN parameters for this channel
        float mean = bn_mean[c];
        float var = bn_var[c];
        float gamma = bn_gamma[c];
        float beta = bn_beta[c];
        
        float val = conv_out[idx];
        
        // Batch Normalization: (x - mean) / sqrt(var + eps) * gamma + beta
        float inv_std = rsqrtf(var + eps);
        val = (val - mean) * inv_std;
        val = val * gamma + beta;
        
        // Add identity
        if (identity != nullptr) {
            val += identity[idx];
        }
        
        // ReLU
        if (val < 0.0f) {
            val = 0.0f;
        }
        
        output[idx] = val;
    }
}

// Kernel for the final layer: Linear + Softmax? Or just Linear?
// The model ends with FC. We can fuse FC if needed, but cuDNN/ATen is good.
// Let's stick to fusing the residual block's tail.

torch::Tensor fused_bn_relu_add_cuda(
    torch::Tensor conv_out,
    torch::Tensor bn_mean,
    torch::Tensor bn_var,
    torch::Tensor bn_gamma,
    torch::Tensor bn_beta,
    torch::Tensor identity,
    float eps
) {
    auto device = conv_out.device();
    auto dtype = conv_out.dtype();
    
    TORCH_CHECK(device.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(conv_out.dim() == 4, "Conv output must be 4D (NCHW)");
    
    int N = conv_out.size(0);
    int C = conv_out.size(1);
    int H = conv_out.size(2);
    int W = conv_out.size(3);
    
    auto output = torch::zeros_like(conv_out);
    
    int total_elements = N * C * H * W;
    if (total_elements == 0) {
        return output;
    }
    
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    // Prepare pointers
    const float* conv_out_ptr = conv_out.data_ptr<float>();
    const float* bn_mean_ptr = bn_mean.data_ptr<float>();
    const float* bn_var_ptr = bn_var.data_ptr<float>();
    const float* bn_gamma_ptr = bn_gamma.data_ptr<float>();
    const float* bn_beta_ptr = bn_beta.data_ptr<float>();
    const float* identity_ptr = identity.numel() > 0 ? identity.data_ptr<float>() : nullptr;
    
    fused_bn_relu_add_kernel<<<num_blocks, block_size>>>(
        conv_out_ptr,
        bn_mean_ptr,
        bn_var_ptr,
        bn_gamma_ptr,
        bn_beta_ptr,
        identity_ptr,
        output.data_ptr<float>(),
        N, C, H, W,
        eps
    );
    
    return output;
}

// Kernel for the first block: Conv1 + BN1 + ReLU (no residual addition initially)
__global__ void fused_conv_bn_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bn_mean,
    const float* __restrict__ bn_var,
    const float* __restrict__ bn_gamma,
    const float* __restrict__ bn_beta,
    float* __restrict__ output,
    int N, int C, int H, int W,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * H * W;
    
    if (idx < total_elements) {
        int temp = idx;
        int w = temp % W;
        temp /= W;
        int h = temp % H;
        temp /= H;
        int c = temp % C;
        
        float mean = bn_mean[c];
        float var = bn_var[c];
        float gamma = bn_gamma[c];
        float beta = bn_beta[c];
        
        float val = input[idx];
        float inv_std = rsqrtf(var + eps);
        val = (val - mean) * inv_std;
        val = val * gamma + beta;
        
        if (val < 0.0f) {
            val = 0.0f;
        }
        
        output[idx] = val;
    }
}

torch::Tensor fused_conv_bn_relu_cuda(
    torch::Tensor input,
    torch::Tensor bn_mean,
    torch::Tensor bn_var,
    torch::Tensor bn_gamma,
    torch::Tensor bn_beta,
    float eps
) {
    auto device = input.device();
    TORCH_CHECK(device.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (NCHW)");
    
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    
    auto output = torch::zeros_like(input);
    
    int total_elements = N * C * H * W;
    if (total_elements == 0) {
        return output;
    }
    
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_conv_bn_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bn_mean.data_ptr<float>(),
        bn_var.data_ptr<float>(),
        bn_gamma.data_ptr<float>(),
        bn_beta.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        eps
    );
    
    return output;
}

"""

custom_ops_cpp_source = (
    "torch::Tensor fused_bn_relu_add_cuda(torch::Tensor conv_out, torch::Tensor bn_mean, torch::Tensor bn_var, torch::Tensor bn_gamma, torch::Tensor bn_beta, torch::Tensor identity, float eps);"
    "torch::Tensor fused_conv_bn_relu_cuda(torch::Tensor input, torch::Tensor bn_mean, torch::Tensor bn_var, torch::Tensor bn_gamma, torch::Tensor bn_beta, float eps);"
);

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=custom_ops_cpp_source,
    cuda_sources=custom_ops_source,
    functions=["fused_bn_relu_add_cuda", "fused_conv_bn_relu_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"]
)


class BasicBlockNew(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlockNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # We will handle ReLU in the custom kernel
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # Conv1 + BN1 + ReLU
        out = self.conv1(x)
        # Fuse BN1 and ReLU
        out = custom_ops.fused_conv_bn_relu_cuda(
            out, 
            self.bn1.running_mean, 
            self.bn1.running_var, 
            self.bn1.weight, 
            self.bn1.bias, 
            self.bn1.eps
        )

        # Conv2 + BN2
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # Fuse Add and ReLU
        out = custom_ops.fused_bn_relu_add_cuda(
            out, 
            self.bn2.running_mean, 
            self.bn2.running_var, 
            self.bn2.weight, 
            self.bn2.bias, 
            identity, 
            self.bn2.eps
        )

        return out


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # ReLU handled in custom kernel
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlockNew, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlockNew, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlockNew, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlockNew, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlockNew.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        # Fuse BN1 and ReLU for the first layer
        x = custom_ops.fused_conv_bn_relu_cuda(
            x, 
            self.bn1.running_mean, 
            self.bn1.running_var, 
            self.bn1.weight, 
            self.bn1.bias, 
            self.bn1.eps
        )
        
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x