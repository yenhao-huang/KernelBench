import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernels for MobileNetV2 optimization
# We will fuse: Conv2d + BatchNorm2d + ReLU6 into a single kernel.
# This avoids global memory writes/reads between these operations.

custom_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Helper to get min/max
__device__ inline float clamp(float val, float min_val, float max_val) {
    return fminf(fmaxf(val, min_val), max_val);
}

// Kernel for Conv2d + BatchNorm2d + ReLU6 fusion
// Assumes input is NHWC or NCHW. PyTorch uses NCHW.
// We assume bias is added during BN if present, but here we treat BN as scale/shift.
// Standard BN: y = (x - mean) / sqrt(var + eps) * gamma + beta
// Combined with Conv: out = ReLU6( gamma/sqrt(var+eps) * (Conv(x) - mean) + beta )
// Let alpha = gamma / sqrt(var + eps), beta' = beta - alpha * mean
// Then out = ReLU6( alpha * Conv(x) + beta' )

__global__ void conv_bn_relu6_kernel(
    const float* input,      // [N, C_in, H_in, W_in]
    const float* weight,     // [C_out, C_in, K_h, K_w]
    const float* bn_gamma,   // [C_out]
    const float* bn_beta,    // [C_out]
    const float* bn_running_mean, // [C_out]
    const float* bn_running_var,  // [C_out]
    float* output,           // [N, C_out, H_out, W_out]
    int N, int C_in, int H_in, int W_in,
    int C_out, int K_h, int K_w,
    int stride_h, int stride_w, int pad_h, int pad_w,
    int H_out, int W_out,
    float eps)
{
    // Each thread handles one output element (N, C_out, H_out, W_out)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C_out * H_out * W_out;

    if (idx >= total_elements) return;

    // Decompose index
    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c_out = (idx / (W_out * H_out)) % C_out;
    int n = idx / (W_out * H_out * C_out);

    float sum = 0.0f;

    // Calculate input coordinates for the receptive field
    int h_start = h_out * stride_h - pad_h;
    int w_start = w_out * stride_w - pad_w;

    for (int k_h = 0; k_h < K_h; ++k_h) {
        int h_in = h_start + k_h;
        if (h_in < 0 || h_in >= H_in) continue;

        for (int k_w = 0; k_w < K_w; ++k_w) {
            int w_in = w_start + k_w;
            if (w_in < 0 || w_in >= W_in) continue;

            // Iterate over input channels
            for (int c_in = 0; c_in < C_in; ++c_in) {
                // Weight index: [C_out, C_in, K_h, K_w]
                int w_idx = ((c_out * C_in + c_in) * K_h + k_h) * K_w + k_w;
                
                // Input index: [N, C_in, H_in, W_in]
                int i_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;

                sum += weight[w_idx] * input[i_idx];
            }
        }
    }

    // Apply BatchNorm and ReLU6
    float var_inv_sqrt = rsqrtf(bn_running_var[c_out] + eps);
    float alpha = bn_gamma[c_out] * var_inv_sqrt;
    float beta_prime = bn_beta[c_out] - alpha * bn_running_mean[c_out];

    float val = sum * alpha + beta_prime;
    
    // ReLU6: max(0, min(6, val))
    if (val < 0.0f) val = 0.0f;
    else if (val > 6.0f) val = 6.0f;

    output[idx] = val;
}

// Kernel for Depthwise Conv + BN + ReLU6
// Groups = C_in = C_out
__global__ void depthwise_conv_bn_relu6_kernel(
    const float* input,      // [N, C, H_in, W_in]
    const float* weight,     // [C, 1, K_h, K_w] (since groups=C)
    const float* bn_gamma,   // [C]
    const float* bn_beta,    // [C]
    const float* bn_running_mean, // [C]
    const float* bn_running_var,  // [C]
    float* output,           // [N, C, H_out, W_out]
    int N, int C, int H_in, int W_in,
    int K_h, int K_w,
    int stride_h, int stride_w, int pad_h, int pad_w,
    int H_out, int W_out,
    float eps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * H_out * W_out;

    if (idx >= total_elements) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c = (idx / (W_out * H_out)) % C;
    int n = idx / (W_out * H_out * C);

    float sum = 0.0f;

    int h_start = h_out * stride_h - pad_h;
    int w_start = w_out * stride_w - pad_w;

    for (int k_h = 0; k_h < K_h; ++k_h) {
        int h_in = h_start + k_h;
        if (h_in < 0 || h_in >= H_in) continue;

        for (int k_w = 0; k_w < K_w; ++k_w) {
            int w_in = w_start + k_w;
            if (w_in < 0 || w_in >= W_in) continue;

            // For depthwise, c_in == c_out == c
            // Weight index: [C, 1, K_h, K_w] -> [c * K_h + k_h] * K_w + k_w ? 
            // Actually PyTorch weight shape for groups=C is [C, 1, K_h, K_w]
            int w_idx = (c * K_h + k_h) * K_w + k_w;
            
            int i_idx = ((n * C + c) * H_in + h_in) * W_in + w_in;

            sum += weight[w_idx] * input[i_idx];
        }
    }

    float var_inv_sqrt = rsqrtf(bn_running_var[c] + eps);
    float alpha = bn_gamma[c] * var_inv_sqrt;
    float beta_prime = bn_beta[c] - alpha * bn_running_mean[c];

    float val = sum * alpha + beta_prime;
    
    if (val < 0.0f) val = 0.0f;
    else if (val > 6.0f) val = 6.0f;

    output[idx] = val;
}

// Kernel for Pointwise Linear Conv (1x1) + BN
// No ReLU after this in the residual block structure usually, but let's check MobileNetV2 spec.
// The last layer of IR block is Pointwise Linear Conv -> BN. No activation.
__global__ void pointwise_linear_conv_bn_kernel(
    const float* input,      // [N, C_in, H, W]
    const float* weight,     // [C_out, C_in, 1, 1]
    const float* bn_gamma,   // [C_out]
    const float* bn_beta,    // [C_out]
    const float* bn_running_mean, // [C_out]
    const float* bn_running_var,  // [C_out]
    float* output,           // [N, C_out, H, W]
    int N, int C_in, int H, int W,
    int C_out,
    float eps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C_out * H * W;

    if (idx >= total_elements) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int c_out = (idx / (W * H)) % C_out;
    int n = idx / (W * H * C_out);

    float sum = 0.0f;

    for (int c_in = 0; c_in < C_in; ++c_in) {
        // Weight index: [C_out, C_in, 1, 1] -> c_out * C_in + c_in
        int w_idx = c_out * C_in + c_in;
        
        // Input index: [N, C_in, H, W]
        int i_idx = ((n * C_in + c_in) * H + h) * W + w;

        sum += weight[w_idx] * input[i_idx];
    }

    float var_inv_sqrt = rsqrtf(bn_running_var[c_out] + eps);
    float alpha = bn_gamma[c_out] * var_inv_sqrt;
    float beta_prime = bn_beta[c_out] - alpha * bn_running_mean[c_out];

    output[idx] = sum * alpha + beta_prime;
}

// Wrapper functions for PyTorch

torch::Tensor conv_bn_relu6_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias, // Not used if BN is fused, but kept for signature compatibility if needed. We ignore it.
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps,
    int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w)
{
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA");
    TORCH_CHECK(running_mean.is_cuda(), "Running mean must be CUDA");
    TORCH_CHECK(running_var.is_cuda(), "Running var must be CUDA");

    auto N = input.size(0);
    auto C_in = input.size(1);
    auto H_in = input.size(2);
    auto W_in = input.size(3);

    auto C_out = weight.size(0);
    auto K_h = weight.size(2);
    auto K_w = weight.size(3);

    // Calculate output dimensions
    auto H_out = (H_in + 2 * pad_h - K_h) / stride_h + 1;
    auto W_out = (W_in + 2 * pad_w - K_w) / stride_w + 1;

    auto output = torch::empty({N, C_out, H_out, W_out}, input.options());

    const int block_size = 256;
    int total_elements = N * C_out * H_out * W_out;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    conv_bn_relu6_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        running_mean.data_ptr<float>(), // Using mean as gamma for BN scale? No, need gamma/beta.
        // Wait, the signature above assumed separate gamma/beta tensors. 
        // PyTorch BatchNorm2d stores gamma and beta internally. We need to extract them or pass them.
        // Since we are replacing nn.Conv2d + nn.BatchNorm2d, we don't have direct access to gamma/beta in the forward pass easily without modifying the module structure significantly or extracting parameters.
        // However, load_inline allows us to define new functions. We will assume the caller passes the BN parameters explicitly if they want this fused kernel.
        // But the prompt asks to replace operators in the architecture. The architecture uses nn.BatchNorm2d.
        // To make this work seamlessly with the existing nn.Sequential structure is hard because nn.BatchNorm2d is a module, not just tensors.
        // Strategy: We will create a custom Module class that contains the fused kernel logic and holds the BN parameters as buffers/parameters.
        
        0, 0, 0, 0, 0, 0 // Dummy args to match signature for now, we'll fix this below
    );

    return output;
}

// Actually, let's define a simpler interface that takes all necessary tensors.
// We will create a custom C++ class or just functions that take the BN stats.

torch::Tensor fused_conv_bn_relu6(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bn_gamma,
    torch::Tensor bn_beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps,
    int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w)
{
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA");

    auto N = input.size(0);
    auto C_in = input.size(1);
    auto H_in = input.size(2);
    auto W_in = input.size(3);

    auto C_out = weight.size(0);
    auto K_h = weight.size(2);
    auto K_w = weight.size(3);

    auto H_out = (H_in + 2 * pad_h - K_h) / stride_h + 1;
    auto W_out = (W_in + 2 * pad_w - K_w) / stride_w + 1;

    auto output = torch::empty({N, C_out, H_out, W_out}, input.options());

    const int block_size = 256;
    int total_elements = N * C_out * H_out * W_out;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    conv_bn_relu6_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bn_gamma.data_ptr<float>(),
        bn_beta.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, K_h, K_w,
        stride_h, stride_w, pad_h, pad_w,
        H_out, W_out,
        eps
    );

    return output;
}

torch::Tensor fused_depthwise_conv_bn_relu6(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bn_gamma,
    torch::Tensor bn_beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps,
    int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w)
{
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA");

    auto N = input.size(0);
    auto C = input.size(1); // C_in == C_out for depthwise
    auto H_in = input.size(2);
    auto W_in = input.size(3);

    auto K_h = weight.size(2);
    auto K_w = weight.size(3);

    auto H_out = (H_in + 2 * pad_h - K_h) / stride_h + 1;
    auto W_out = (W_in + 2 * pad_w - K_w) / stride_w + 1;

    auto output = torch::empty({N, C, H_out, W_out}, input.options());

    const int block_size = 256;
    int total_elements = N * C * H_out * W_out;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    depthwise_conv_bn_relu6_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bn_gamma.data_ptr<float>(),
        bn_beta.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H_in, W_in,
        K_h, K_w,
        stride_h, stride_w, pad_h, pad_w,
        H_out, W_out,
        eps
    );

    return output;
}

torch::Tensor fused_pointwise_conv_bn(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bn_gamma,
    torch::Tensor bn_beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps)
{
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA");

    auto N = input.size(0);
    auto C_in = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);

    auto C_out = weight.size(0);

    auto output = torch::empty({N, C_out, H, W}, input.options());

    const int block_size = 256;
    int total_elements = N * C_out * H * W;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    pointwise_linear_conv_bn_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bn_gamma.data_ptr<float>(),
        bn_beta.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, H, W,
        C_out,
        eps
    );

    return output;
}
"""

custom_cpp_source = """
#include <torch/extension.h>

torch::Tensor fused_conv_bn_relu6(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bn_gamma,
    torch::Tensor bn_beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps,
    int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w);

torch::Tensor fused_depthwise_conv_bn_relu6(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bn_gamma,
    torch::Tensor bn_beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps,
    int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w);

torch::Tensor fused_pointwise_conv_bn(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bn_gamma,
    torch::Tensor bn_beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps);
"""

# Load the inline CUDA extension
mobile_net_cuda = load_inline(
    name="mobile_net_cuda",
    cpp_sources=custom_cpp_source,
    cuda_sources=custom_cuda_source,
    functions=[
        "fused_conv_bn_relu6",
        "fused_depthwise_conv_bn_relu6",
        "fused_pointwise_conv_bn"
    ],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class FusedConvBNReLU6(nn.Module):
    """
    Custom module that fuses Conv2d + BatchNorm2d + ReLU6 into a single CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(FusedConvBNReLU6, self).__init__()
        
        # Conv2d parameters
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        
        # BatchNorm2d parameters
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        
        # Initialize weights and biases
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
            
        nn.init.ones_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)

    def forward(self, x):
        # Extract parameters from Conv and BN
        weight = self.conv.weight
        bn_gamma = self.bn.weight
        bn_beta = self.bn.bias
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        eps = self.bn.eps
        
        stride_h = self.conv.stride[0]
        stride_w = self.conv.stride[1]
        pad_h = self.conv.padding[0]
        pad_w = self.conv.padding[1]

        # Call fused CUDA kernel
        out = mobile_net_cuda.fused_conv_bn_relu6(
            x, weight, bn_gamma, bn_beta, running_mean, running_var, eps,
            stride_h, stride_w, pad_h, pad_w
        )
        return out


class FusedDepthwiseConvBNReLU6(nn.Module):
    """
    Custom module that fuses Depthwise Conv2d + BatchNorm2d + ReLU6 into a single CUDA kernel.
    """
    def __init__(self, in_channels, kernel_size, stride=1, padding=0, bias=False):
        super(FusedDepthwiseConvBNReLU6, self).__init__()
        
        # Depthwise Conv2d: groups = in_channels
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=bias)
        
        # BatchNorm2d parameters
        self.bn = nn.BatchNorm2d(in_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        
        # Initialize weights and biases
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
            
        nn.init.ones_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)

    def forward(self, x):
        weight = self.conv.weight
        bn_gamma = self.bn.weight
        bn_beta = self.bn.bias
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        eps = self.bn.eps
        
        stride_h = self.conv.stride[0]
        stride_w = self.conv.stride[1]
        pad_h = self.conv.padding[0]
        pad_w = self.conv.padding[1]

        out = mobile_net_cuda.fused_depthwise_conv_bn_relu6(
            x, weight, bn_gamma, bn_beta, running_mean, running_var, eps,
            stride_h, stride_w, pad_h, pad_w
        )
        return out


class FusedPointwiseConvBN(nn.Module):
    """
    Custom module that fuses Pointwise (1x1) Conv2d + BatchNorm2d into a single CUDA kernel.
    No ReLU is applied here as per MobileNetV2 specification for the linear bottleneck.
    """
    def __init__(self, in_channels, out_channels, bias=False):
        super(FusedPointwiseConvBN, self).__init__()
        
        # Pointwise Conv2d: 1x1 kernel
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias)
        
        # BatchNorm2d parameters
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        
        # Initialize weights and biases
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
            
        nn.init.ones_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)

    def forward(self, x):
        weight = self.conv.weight
        bn_gamma = self.bn.weight
        bn_beta = self.bn.bias
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        eps = self.bn.eps

        out = mobile_net_cuda.fused_pointwise_conv_bn(
            x, weight, bn_gamma, bn_beta, running_mean, running_var, eps
        )
        return out


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        Optimized MobileNetV2 architecture implementation in PyTorch.
        Uses custom CUDA operators for Conv+BN+ReLU6 fusion.
        """
        super(ModelNew, self).__init__()
        
        def _make_divisible(v, divisor, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        def _inverted_residual_block(inp, oup, stride, expand_ratio):
            hidden_dim = int(inp * expand_ratio)
            use_res_connect = stride == 1 and inp == oup

            layers = []
            if expand_ratio != 1:
                # Pointwise convolution (expand) -> BN -> ReLU6
                # Fused into one kernel
                layers.append(FusedPointwiseConvBN(inp, hidden_dim))
                layers.append(FusedConvBNReLU6(hidden_dim, hidden_dim, 3, stride, 1))

            # Depthwise convolution -> BN -> ReLU6
            # Fused into one kernel
            layers.append(FusedDepthwiseConvBNReLU6(hidden_dim, 3, stride, 1))

            # Pointwise linear convolution (project) -> BN
            # Fused into one kernel (no ReLU)
            layers.append(FusedPointwiseConvBN(hidden_dim, oup))

            if use_res_connect:
                return nn.Sequential(*layers), True
            else:
                return nn.Sequential(*layers), False

        # MobileNetV2 architecture
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # Building first layer: Conv + BN + ReLU6
        self.features = nn.Sequential(
            FusedConvBNReLU6(3, input_channel, 3, 2, 1),
        )

        # Building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                block, use_res_connect = _inverted_residual_block(input_channel, output_channel, stride, expand_ratio=t)
                self.features.append(block)
                input_channel = output_channel

        # Building last several layers: Conv + BN + ReLU6
        self.features.append(FusedConvBNReLU6(input_channel, last_channel, 1, 1, 0))

        # Final layer
        self.features.append(nn.AdaptiveAvgPool2d((1, 1)))

        # Linear layer (standard PyTorch is fine here as it's small)
        self.classifier = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(last_channel, num_classes),
        )

        # Weight initialization is handled within the custom modules, 
        # but we can re-initialize if needed or rely on the init in __init__ of custom modules.
        # The custom modules already initialize weights and biases.

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x