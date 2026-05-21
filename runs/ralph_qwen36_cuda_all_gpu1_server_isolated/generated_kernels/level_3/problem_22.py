import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernels for fused operations to optimize EfficientNetB0

# 1. Fused Conv2d + BatchNorm2d + ReLU6 (or ReLU)
# This kernel performs: out = clamp( (x * W).sum(dim=...) * gamma + beta, 0, 6 )
# Note: For standard ReLU in the first layer, we can use a similar kernel with max(0, x)
fused_conv_bn_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Helper for ReLU6
__device__ __forceinline__ float relu6(float x) {
    return fminf(fmaxf(x, 0.0f), 6.0f);
}

// Helper for standard ReLU
__device__ __forceinline__ float relu(float x) {
    return fmaxf(x, 0.0f);
}

struct ConvBNParams {
    const float* weight; // [out_channels, in_channels, kH, kW]
    const float* bias;   // [out_channels] (optional, can be null if fused into BN)
    const float* gamma;  // [out_channels] from BatchNorm
    const float* beta;   // [out_channels] from BatchNorm
    const float* mean;   // [out_channels] from BatchNorm
    const float* var;    // [out_channels] from BatchNorm
    int in_channels;
    int out_channels;
    int height;
    int width;
    int kernel_h;
    int kernel_w;
    int stride_h;
    int stride_w;
    int pad_h;
    int pad_w;
    int dilation_h;
    int dilation_w;
    bool use_relu6;
};

__global__ void fused_conv_bn_relu_kernel(const float* input, ConvBNParams params, float* output) {
    // Each thread handles one output element (batch, out_channel, h, w)
    // We assume batch size is 1 for simplicity in this specific optimization context 
    // or handle it via grid stride loops. EfficientNet usually processes batches.
    // To support arbitrary batch sizes, we map blockIdx to spatial locations and use grid-stride for batch/channels if needed,
    // but a simpler approach for high performance on typical inference/training shapes is:
    // One thread per output pixel (b, c, h, w).
    
    int total_elements = params.out_channels * params.height * params.width; 
    // Note: This kernel assumes batch_size=1 for the grid mapping simplicity. 
    // For general batch support, we would need to include batch index in the calculation.
    // Let's implement a version that supports batch size > 1 by treating the first dimension as part of the loop or grid.
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = params.height * params.width * params.out_channels;
    
    if (idx >= total_size) return;

    // Map linear index to (out_channel, h, w)
    int c_out = idx % params.out_channels;
    int hw_idx = idx / params.out_channels;
    int w = hw_idx % params.width;
    int h = (hw_idx / params.width) % params.height;

    float sum = 0.0f;
    
    // Iterate over input channels and kernel dimensions
    for (int c_in = 0; c_in < params.in_channels; ++c_in) {
        for (int kh = 0; kh < params.kernel_h; ++kh) {
            for (int kw = 0; kw < params.kernel_w; ++kw) {
                int in_h = h * params.stride_h - params.pad_h + kh * params.dilation_h;
                int in_w = w * params.stride_w - params.pad_w + kw * params.dilation_w;
                
                if (in_h >= 0 && in_h < params.height + 2 * params.pad_h && 
                    in_w >= 0 && in_w < params.width + 2 * params.pad_w) { // Simplified padding check logic usually requires explicit input size handling or padding insertion. 
                    // For standard Conv2d with padding, the input tensor passed to this kernel should ideally be padded or we calculate indices carefully.
                    // PyTorch's conv2d handles padding internally. To fuse it efficiently without pre-padding, we need to check bounds against the *padded* logical space.
                    // However, accessing out-of-bounds memory is bad. 
                    // Standard optimization: Pad input tensor before calling this kernel, or use a kernel that checks bounds against original H/W and applies padding logic.
                    // Given the complexity of implementing full padding logic in inline CUDA without pre-padding, 
                    // and since PyTorch's conv2d is highly optimized (cuDNN), fusing Conv+BN+ReLU is often less beneficial than just BN+ReLU if Conv is already fast.
                    // BUT, for MBConv depthwise and 1x1 convs, custom kernels can help.
                    
                    // Let's assume the input tensor 'input' has been padded by the caller or we handle padding via index calculation relative to original H/W.
                    // Actually, let's stick to a simpler fusion: BatchNorm + ReLU6 on top of existing Conv outputs, 
                    // OR fuse 1x1 Conv + BN + ReLU6 which is very common in MBConv expand/project.
                    
                    // To make this robust and compilable without complex padding logic in the kernel:
                    // We will implement a fused Depthwise Conv + BN + ReLU6 and Fused 1x1 Conv + BN + ReLU6 separately if needed, 
                    // or just fuse BN+ReLU6 which is universally applicable and safe.
                    
                    // Let's refine the strategy: 
                    // 1. Fuse BatchNorm2d + ReLU6 (or ReLU) into a single kernel. This is always safe and effective.
                    // 2. For MBConv, the bottleneck is often the sequence of small convs. 
                    //    Fusing BN+ReLU reduces memory bandwidth for intermediate activations.
                }
            }
        }
    }
    
    // Since implementing full padding logic in a single inline kernel is error-prone and verbose,
    // we will focus on fusing BatchNorm2d + Activation (ReLU/ReLU6) which is universally applicable 
    // and removes the need to write intermediate tensors to global memory.
}

// Kernel for Fused BatchNorm2d + ReLU6
__global__ void fused_bn_relu6_kernel(const float* input, const float* gamma, const float* beta, const float* mean, const float* var_inv_sqrt, int channels, int height, int width, float* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = channels * height * width;
    
    if (idx < total_elements) {
        // Calculate channel index
        int c = idx % channels;
        
        // Load parameters for this channel
        float g = gamma[c];
        float b = beta[c];
        float m = mean[c];
        float v_inv_sqrt = var_inv_sqrt[c];
        
        // Load input value (spatial location doesn't matter for BN stats, but we need to read the pixel)
        // The input is laid out as [N, C, H, W]. 
        // We can iterate over N, H, W in the grid or use a 1D index.
        // Let's assume N=1 for simplicity in this specific kernel design to keep it compact, 
        // or handle N by including it in the total size calculation.
        
        // Correct mapping for [N, C, H, W] with 1D grid:
        int n = idx / (channels * height * width); // This logic is flawed if we just use linear index.
        
        // Let's restart the indexing for a general N, C, H, W tensor.
        // Total size = N * C * H * W.
        // We want to process all elements.
    }
}

// Better approach: 3D Grid mapping for [N, C, H, W]
__global__ void fused_bn_relu6_kernel_v2(const float* input, const float* gamma, const float* beta, const float* mean, const float* var_inv_sqrt, int N, int C, int H, int W, float* output) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (w < W && h < H && c < C) {
        // We need to iterate over N. 
        // To support arbitrary N, we can use a grid-stride loop for N or assume N=1.
        // For EfficientNet inference/training, N varies. 
        // Let's implement a kernel that handles N by having the caller launch it with N blocks in Z or using a loop.
        // Simpler: Launch kernel with grid covering C, H, W and use a loop for N? No, that's slow.
        // Best: Map (n, c, h, w) to threads. 
        // If we fix the launch configuration to cover C, H, W, we still need to handle N.
        
        // Let's assume the input tensor is contiguous and we process all elements.
        // We will use a 1D grid covering N*C*H*W for simplicity in this inline example, 
        // but optimize memory access by coalescing.
    }
}

// Final Decision: Implement Fused BatchNorm2d + ReLU6 (and ReLU) kernel that handles arbitrary batch sizes efficiently.
// We will use a 1D grid covering the entire spatial+channel dimension for a single batch, 
// and launch N times? No, that's inefficient.
// We will map blockIdx to (N, C, H, W) using multiple dimensions if possible, or just 1D with stride.

__global__ void fused_bn_relu_kernel(const float* input, const float* gamma, const float* beta, const float* mean, const float* var_inv_sqrt, int N, int C, int H, int W, bool is_relu6, float* output) {
    // Total elements per batch: C * H * W
    int total_spatial = C * H * W;
    
    // Global index across all batches
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * total_spatial;
    
    if (idx >= total_elements) return;
    
    // Determine batch, channel, and spatial index
    int n = idx / total_spatial;
    int spatial_idx = idx % total_spatial;
    
    int c = spatial_idx % C;
    int hw = spatial_idx / C;
    int w = hw % W;
    int h = (hw / W) % H;
    
    // Load input value
    float val = input[idx]; // Coalesced access if idx is linear
    
    // Apply BatchNorm
    float g = gamma[c];
    float b = beta[c];
    float m = mean[c];
    float v_inv_sqrt = var_inv_sqrt[c];
    
    float normalized = (val - m) * v_inv_sqrt;
    float result = normalized * g + b;
    
    // Apply Activation
    if (is_relu6) {
        result = fminf(fmaxf(result, 0.0f), 6.0f);
    } else {
        result = fmaxf(result, 0.0f);
    }
    
    output[idx] = result;
}

// Kernel for Fused Depthwise Conv + BN + ReLU6
// This is specific to MBConv depthwise layers.
__global__ void fused_dw_conv_bn_relu6_kernel(const float* input, const float* weight, const float* gamma, const float* beta, const float* mean, const float* var_inv_sqrt, int N, int C, int H, int W, int kernel_size, int stride, int pad, float* output) {
    // Each thread handles one output pixel (n, c, h, w)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_spatial = C * H * W;
    int total_elements = N * total_spatial;
    
    if (idx >= total_elements) return;
    
    int n = idx / total_spatial;
    int spatial_idx = idx % total_spatial;
    int c = spatial_idx % C;
    int hw = spatial_idx / C;
    int w = hw % W;
    int h = (hw / W) % H;
    
    float sum = 0.0f;
    
    // Depthwise convolution: each channel is convolved independently with its own kernel
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int in_h = h * stride - pad + kh;
            int in_w = w * stride - pad + kw;
            
            if (in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
                // Input index for this spatial location and channel
                int input_idx = n * total_spatial + c * H * W + in_h * W + in_w;
                
                // Weight index: [C, kernel_size, kernel_size]
                int weight_idx = c * kernel_size * kernel_size + kh * kernel_size + kw;
                
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    
    // Apply BatchNorm and ReLU6
    float g = gamma[c];
    float b = beta[c];
    float m = mean[c];
    float v_inv_sqrt = var_inv_sqrt[c];
    
    float result = (sum - m) * v_inv_sqrt * g + b;
    result = fminf(fmaxf(result, 0.0f), 6.0f);
    
    output[idx] = result;
}

// Kernel for Fused 1x1 Conv + BN + ReLU6 (Expand/Project)
__global__ void fused_1x1_conv_bn_relu6_kernel(const float* input, const float* weight, const float* gamma, const float* beta, const float* mean, const float* var_inv_sqrt, int N, int C_in, int C_out, int H, int W, float* output) {
    // Each thread handles one output pixel (n, c_out, h, w)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_spatial = C_out * H * W;
    int total_elements = N * total_spatial;
    
    if (idx >= total_elements) return;
    
    int n = idx / total_spatial;
    int spatial_idx = idx % total_spatial;
    int c_out = spatial_idx % C_out;
    int hw = spatial_idx / C_out;
    int w = hw % W;
    int h = (hw / W) % H;
    
    float sum = 0.0f;
    
    // 1x1 Conv: sum over input channels
    for (int c_in = 0; c_in < C_in; ++c_in) {
        int input_idx = n * (C_in * H * W) + c_in * H * W + h * W + w;
        // Weight is [C_out, C_in] for 1x1 conv
        int weight_idx = c_out * C_in + c_in;
        
        sum += input[input_idx] * weight[weight_idx];
    }
    
    // Apply BatchNorm and ReLU6
    float g = gamma[c_out];
    float b = beta[c_out];
    float m = mean[c_out];
    float v_inv_sqrt = var_inv_sqrt[c_out];
    
    float result = (sum - m) * v_inv_sqrt * g + b;
    result = fminf(fmaxf(result, 0.0f), 6.0f);
    
    output[idx] = result;
}

// Host functions to launch kernels
torch::Tensor fused_bn_relu_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, torch::Tensor mean, torch::Tensor var_inv_sqrt, bool is_relu6) {
    auto output = torch::empty_like(input);
    
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    
    const int block_size = 256;
    const int total_elements = N * C * H * W;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_bn_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        mean.data_ptr<float>(),
        var_inv_sqrt.data_ptr<float>(),
        N, C, H, W, is_relu6,
        output.data_ptr<float>()
    );
    
    return output;
}

torch::Tensor fused_dw_conv_bn_relu6_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor gamma, torch::Tensor beta, torch::Tensor mean, torch::Tensor var_inv_sqrt, int kernel_size, int stride, int pad) {
    auto output = torch::empty_like(input);
    
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    
    const int block_size = 256;
    const int total_elements = N * C * H * W;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_dw_conv_bn_relu6_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        mean.data_ptr<float>(),
        var_inv_sqrt.data_ptr<float>(),
        N, C, H, W, kernel_size, stride, pad,
        output.data_ptr<float>()
    );
    
    return output;
}

torch::Tensor fused_1x1_conv_bn_relu6_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor gamma, torch::Tensor beta, torch::Tensor mean, torch::Tensor var_inv_sqrt) {
    auto output = torch::empty_like(input);
    
    int N = input.size(0);
    int C_in = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int C_out = weight.size(0); // [C_out, C_in]
    
    const int block_size = 256;
    const int total_elements = N * C_out * H * W;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_1x1_conv_bn_relu6_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        mean.data_ptr<float>(),
        var_inv_sqrt.data_ptr<float>(),
        N, C_in, C_out, H, W,
        output.data_ptr<float>()
    );
    
    return output;
}

"""

fused_bn_relu_cpp_source = (
    "torch::Tensor fused_bn_relu_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, torch::Tensor mean, torch::Tensor var_inv_sqrt, bool is_relu6);"
    "torch::Tensor fused_dw_conv_bn_relu6_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor gamma, torch::Tensor beta, torch::Tensor mean, torch::Tensor var_inv_sqrt, int kernel_size, int stride, int pad);"
    "torch::Tensor fused_1x1_conv_bn_relu6_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor gamma, torch::Tensor beta, torch::Tensor mean, torch::Tensor var_inv_sqrt);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_bn_relu_cpp_source,
    cuda_sources=fused_bn_relu_source,
    functions=["fused_bn_relu_cuda", "fused_dw_conv_bn_relu6_cuda", "fused_1x1_conv_bn_relu6_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class FusedBNReLU(nn.Module):
    def __init__(self, num_features, is_relu6=True):
        super(FusedBNReLU, self).__init__()
        self.num_features = num_features
        self.is_relu6 = is_relu6
        # Initialize parameters that will be registered as buffers or parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        # Calculate var_inv_sqrt from running_var
        # Add eps to avoid division by zero
        var_inv_sqrt = 1.0 / torch.sqrt(self.running_var + 1e-5)
        
        # Use custom CUDA kernel
        return fused_ops.fused_bn_relu_cuda(x, self.gamma, self.beta, self.running_mean, var_inv_sqrt, self.is_relu6)


class FusedDWConvBNReLU6(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, pad):
        super(FusedDWConvBNReLU6, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        
        # Initialize weights for depthwise conv: [C, 1, K, K] -> flattened to [C, K*K] for the kernel
        # PyTorch Conv2d groups=in_channels means weight shape is [out_channels, in_channels/groups, kH, kW]
        # Here out_channels = in_channels, groups = in_channels.
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size))
        
        # BN parameters
        self.bn = FusedBNReLU(in_channels, is_relu6=True)

    def forward(self, x):
        # Extract BN stats from the internal FusedBNReLU module
        gamma = self.bn.gamma
        beta = self.bn.beta
        mean = self.bn.running_mean
        var_inv_sqrt = 1.0 / torch.sqrt(self.bn.running_var + 1e-5)
        
        # Flatten weight for the kernel: [C, K*K]
        weight_flat = self.weight.view(self.in_channels, -1)
        
        return fused_ops.fused_dw_conv_bn_relu6_cuda(
            x, 
            weight_flat, 
            gamma, 
            beta, 
            mean, 
            var_inv_sqrt, 
            self.kernel_size, 
            self.stride, 
            self.pad
        )


class Fused1x1ConvBNReLU6(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Fused1x1ConvBNReLU6, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Initialize weights for 1x1 conv: [out_channels, in_channels]
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels))
        
        # BN parameters
        self.bn = FusedBNReLU(out_channels, is_relu6=True)

    def forward(self, x):
        gamma = self.bn.gamma
        beta = self.bn.beta
        mean = self.bn.running_mean
        var_inv_sqrt = 1.0 / torch.sqrt(self.bn.running_var + 1e-5)
        
        return fused_ops.fused_1x1_conv_bn_relu6_cuda(
            x, 
            self.weight, 
            gamma, 
            beta, 
            mean, 
            var_inv_sqrt
        )


class MBConvNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(MBConvNew, self).__init__()
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio
        
        if expand_ratio != 1:
            # Expand: 1x1 Conv + BN + ReLU6
            self.expand_conv = Fused1x1ConvBNReLU6(in_channels, hidden_dim)
        
        # Depthwise: DW Conv + BN + ReLU6
        pad = (kernel_size - 1) // 2
        self.depthwise_conv = FusedDWConvBNReLU6(hidden_dim, kernel_size, stride, pad)
        
        # Project: 1x1 Conv + BN (no activation after project in MBConv usually, or linear)
        # The original code has no ReLU after project. We fuse Conv+BN only?
        # Our current fused kernels include ReLU6. 
        # For the project layer, we can use a custom kernel without activation or just call BN separately.
        # To keep it simple and consistent with the "fused" theme where possible:
        # If there's no activation, we can't use the ReLU6 fused kernels directly.
        # However, for speedup, we can fuse Conv+BN. Let's create a simple Conv+BN kernel or just use standard layers if no activation is needed.
        # Given the constraints and to ensure correctness, let's use standard Conv2d + BatchNorm2d for the project layer 
        # because fusing Conv+BN without activation is less common in these specific inline examples and might not yield as much benefit 
        # compared to the memory bandwidth savings of BN+ReLU fusion.
        # BUT, we can still fuse it if we want. Let's stick to standard layers for the project part to avoid complexity, 
        # or implement a fused Conv+BN kernel.
        # For the sake of this exercise, let's use standard layers for the project layer as it's the last step in the block and often small.
        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x
        
        if hasattr(self, 'expand_conv'):
            x = self.expand_conv(x)
        
        x = self.depthwise_conv(x)
        x = self.project_conv(x)
        
        if self.use_residual:
            x += identity
        
        return x


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        Optimized EfficientNetB0 architecture implementation in PyTorch.
        Uses custom CUDA operators for fused BatchNorm+ReLU operations.
        """
        super(ModelNew, self).__init__()
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        # Fuse BN + ReLU (standard ReLU, not ReLU6) for the first layer
        self.bn1_relu = FusedBNReLU(32, is_relu6=False)
        
        # MBConv blocks
        self.blocks = nn.Sequential(
            # MBConv1 (32, 16, 1, 1)
            MBConvNew(32, 16, kernel_size=3, stride=1, expand_ratio=1),
            # MBConv6 (16, 24, 2, 6)
            MBConvNew(16, 24, kernel_size=3, stride=2, expand_ratio=6),
            # MBConv6 (24, 24, 1, 6)
            MBConvNew(24, 24, kernel_size=3, stride=1, expand_ratio=6),
            # MBConv6 (24, 40, 2, 6)
            MBConvNew(24, 40, kernel_size=5, stride=2, expand_ratio=6),
            # MBConv6 (40, 40, 1, 6)
            MBConvNew(40, 40, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (40, 80, 2, 6)
            MBConvNew(40, 80, kernel_size=3, stride=2, expand_ratio=6),
            # MBConv6 (80, 80, 1, 6)
            MBConvNew(80, 80, kernel_size=3, stride=1, expand_ratio=6),
            # MBConv6 (80, 112, 1, 6)
            MBConvNew(80, 112, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (112, 112, 1, 6)
            MBConvNew(112, 112, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (112, 192, 2, 6)
            MBConvNew(112, 192, kernel_size=5, stride=2, expand_ratio=6),
            # MBConv6 (192, 192, 1, 6)
            MBConvNew(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (192, 192, 1, 6)
            MBConvNew(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (192, 320, 1, 6)
            MBConvNew(192, 320, kernel_size=3, stride=1, expand_ratio=6)
        )
        
        # Final convolutional layer
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        # Fuse BN + ReLU (standard ReLU) for the final layer before pooling
        self.bn2_relu = FusedBNReLU(1280, is_relu6=False)
        
        # Fully connected layer
        self.fc = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        """
        Forward pass of the Optimized EfficientNetB0 model.
        """
        x = self.conv1(x)
        x = self.bn1_relu(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.bn2_relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Test code placeholders to ensure structure matches requirements
batch_size = 10
num_classes = 1000

def get_inputs():
    return [torch.rand(batch_size, 3, 224, 224)]

def get_init_inputs():
    return [num_classes]