import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernels for Conv2d + ReLU fusion and Element-wise Add/ReLU
# We will fuse the Convolution with the subsequent ReLU activation to reduce memory traffic.
# We also optimize the FireModule's expand layers by fusing Conv+ReLU.

custom_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
        } \
    } while (0)

// Kernel for Conv2d + ReLU
// Assumes NHWC layout internally for better coalescing if possible, but standard NCHW is used here.
// To optimize NCHW convolution, we typically use im2col or direct convolution. 
// Given the constraints of inline code and simplicity vs performance trade-off, 
// we will implement a tiled direct convolution with ReLU fusion.

__global__ void conv2d_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int height,
    int width,
    int out_channels,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w
) {
    // Grid-stride loop for output pixels
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out_pixels = batch_size * out_channels * height * width;

    if (out_idx >= total_out_pixels) return;

    // Decompose linear index into coordinates
    int w = out_idx % width;
    int temp = out_idx / width;
    int h = temp % height;
    temp /= height;
    int c = temp % out_channels;
    int b = temp / out_channels;

    float sum = 0.0f;
    
    // Iterate over input channels and kernel dimensions
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int ih = h * stride_h - pad_h + kh * dilation_h;
                int iw = w * stride_w - pad_w + kw * dilation_w;

                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    // Input index: NCHW
                    int input_idx = ((b * in_channels + ic) * height + ih) * width + iw;
                    // Weight index: OIHW
                    int weight_idx = ((c * in_channels + ic) * kernel_h + kh) * kernel_w + kw;
                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[c];
    }

    // Apply ReLU
    if (sum < 0.0f) sum = 0.0f;

    int output_idx = ((b * out_channels + c) * height + h) * width + w;
    output[output_idx] = sum;
}

// Optimized Conv2d + ReLU using a more efficient memory access pattern would be ideal,
// but for inline simplicity and correctness without external libraries like CUTLASS,
// we provide a standard implementation. Note: For production, cuDNN is preferred.
// However, to demonstrate "custom operator" capability and potential fusion benefits 
// (avoiding intermediate tensor allocation), we implement this fused kernel.

torch::Tensor conv2d_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda() || bias.numel() == 0, "Bias must be a CUDA tensor or empty");

    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);

    auto out_channels = weight.size(0);
    auto kernel_h = weight.size(2);
    auto kernel_w = weight.size(3);

    // Assume stride 1, padding 0 for simplicity in this generic fused kernel 
    // or derive from weight shape if we assume standard conv. 
    // The model uses specific strides/pads. We need to pass them or infer.
    // For this example, we assume stride=1, pad=0, dilation=1 as a baseline 
    // and adjust the kernel call parameters. 
    // To make it robust for the given architecture (which has various pads/strides),
    // we will hardcode the logic for specific layers or pass params.
    // Let's assume stride 1, pad 0 for the generic kernel above and create specialized wrappers 
    // or just use a more general kernel. 
    
    // Actually, let's write a more general kernel that takes strides/pads as args.
    // But to keep code size manageable, we will implement a specific version for 
    // the common cases in SqueezeNet (FireModule).
    
    // FireModule uses:
    // 1x1 convs: stride 1, pad 0
    // 3x3 convs: stride 1, pad 1
    
    // We will create two kernels or one flexible one. Let's make one flexible one.
    
    auto output = torch::empty({batch_size, out_channels, height, width}, input.options());

    const int block_size = 256;
    int total_out_pixels = batch_size * out_channels * height * width;
    int num_blocks = (total_out_pixels + block_size - 1) / block_size;

    // We need to pass stride/pad/dilation. 
    // Since the kernel signature above is fixed, let's redefine the kernel call below with specific params 
    // or modify the kernel to accept them.
    
    // Redefining kernel inside the source string is not possible in C++ easily without templates/macros.
    // We will use a macro-based approach or just assume standard parameters for the sake of this exercise 
    // and create specific kernels for 1x1 and 3x3.
    
    return output;
}

// Let's restart the kernel definition to be more robust and include stride/pad/dilation args.

__global__ void conv2d_relu_general_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int height,
    int width,
    int out_channels,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out_pixels = batch_size * out_channels * height * width;

    if (out_idx >= total_out_pixels) return;

    int w = out_idx % width;
    int temp = out_idx / width;
    int h = temp % height;
    temp /= height;
    int c = temp % out_channels;
    int b = temp / out_channels;

    float sum = 0.0f;
    
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int ih = h * stride_h - pad_h + kh * dilation_h;
                int iw = w * stride_w - pad_w + kw * dilation_w;

                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    int input_idx = ((b * in_channels + ic) * height + ih) * width + iw;
                    int weight_idx = ((c * in_channels + ic) * kernel_h + kh) * kernel_w + kw;
                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    if (bias != nullptr && bias.numel() > 0) {
        sum += bias[c];
    }

    if (sum < 0.0f) sum = 0.0f;

    int output_idx = ((b * out_channels + c) * height + h) * width + w;
    output[output_idx] = sum;
}

torch::Tensor conv2d_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");

    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);

    auto out_channels = weight.size(0);
    auto kernel_h = weight.size(2);
    auto kernel_w = weight.size(3);

    auto output = torch::empty({batch_size, out_channels, height, width}, input.options());

    const int block_size = 256;
    int total_out_pixels = batch_size * out_channels * height * width;
    int num_blocks = (total_out_pixels + block_size - 1) / block_size;

    if (num_blocks == 0) return output;

    conv2d_relu_general_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.numel() > 0 ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, height, width, out_channels,
        kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w
    );

    CUDA_CHECK(cudaGetLastError());
    return output;
}

// Kernel for FireModule Concatenation + ReLU is not strictly necessary as Cat is fast, 
// but we can fuse the two expand branches if they share the same input.
// However, standard Cat is very optimized. We will stick to fusing Conv+ReLU.

"""

custom_cpp_source = """
#include <torch/extension.h>

torch::Tensor conv2d_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_relu_cuda", &conv2d_relu_cuda, "Conv2d + ReLU CUDA kernel");
}
"""

# Load the custom extension
custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=custom_cpp_source,
    cuda_sources=custom_cuda_source,
    functions=["conv2d_relu_cuda"],
    verbose=False,
)


class FusedConv2dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(FusedConv2dReLU, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Initialize weights and bias manually to match PyTorch defaults for consistency
        # Note: In a real scenario, we would load pre-trained weights. 
        # Here we assume the model is trained with standard Conv2d, so we need to replicate that behavior.
        # However, since we are replacing the operator in an existing architecture, 
        # we must ensure the weights are compatible.
        
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=0.05)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Extract parameters for the CUDA kernel
        stride_h, stride_w = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        pad_h, pad_w = self.padding if isinstance(self.padding, tuple) else (self.padding, self.padding)
        dilation_h, dilation_w = self.dilation if isinstance(self.dilation, tuple) else (self.dilation, self.dilation)
        
        # Handle kernel size for non-square kernels if necessary, but FireModule uses 1x1 and 3x3
        kh, kw = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size, self.kernel_size)
        
        return custom_ops.conv2d_relu_cuda(
            x, 
            self.weight, 
            self.bias, 
            stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w
        )

import math

class FireModuleNew(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(FireModuleNew, self).__init__()
        
        # Replace standard Conv2d+ReLU with fused custom operator
        self.squeeze = FusedConv2dReLU(in_channels, squeeze_channels, kernel_size=1)
        
        self.expand1x1 = FusedConv2dReLU(squeeze_channels, expand1x1_channels, kernel_size=1)
        
        self.expand3x3 = FusedConv2dReLU(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.squeeze(x)
        return torch.cat([
            self.expand1x1(x),
            self.expand3x3(x)
        ], 1)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        
        # The first layer is Conv2d(3, 96, kernel_size=7, stride=2) followed by ReLU
        # We can fuse this too.
        self.features = nn.Sequential(
            FusedConv2dReLU(3, 96, kernel_size=7, stride=2),
            # MaxPool2d is left as is because it's already highly optimized in cuDNN/CUDA
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModuleNew(96, 16, 64, 64),
            FireModuleNew(128, 16, 64, 64),
            FireModuleNew(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModuleNew(256, 32, 128, 128),
            FireModuleNew(256, 48, 192, 192),
            FireModuleNew(384, 48, 192, 192),
            FireModuleNew(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModuleNew(512, 64, 256, 256),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.0),
            # The last conv layer in classifier is also Conv+ReLU
            FusedConv2dReLU(512, num_classes, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)

# Note: To use this model with pre-trained weights from the original Model, 
# you would need to copy the weights from the original nn.Conv2d layers into the 
# FusedConv2dReLU.weight and .bias parameters. The initialization above is random.