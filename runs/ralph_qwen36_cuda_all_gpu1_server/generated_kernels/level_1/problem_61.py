import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA implementation for Transposed 3D Convolution (ConvTranspose3d)
# This kernel performs the operation: out = input * weight (transposed) + bias
# It assumes square kernels and specific strides/paddings for optimization, 
# but handles general cases via index calculation.
# Optimization strategy: Use shared memory for weights if possible, or optimized global memory access patterns.
# For simplicity and robustness in a single kernel without complex tiling logic for arbitrary 3D sizes,
# we will implement a direct mapping where each output element is computed by iterating over the receptive field.
# To maximize speed, we use a 1D grid of threads where each thread computes one output element.

conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Helper to calculate linear index from 5D coordinates (N, C, D, H, W)
__device__ inline int idx(int n, int c, int d, int h, int w, int stride_n, int stride_c, int stride_d, int stride_h, int stride_w) {
    return ((n * stride_c + c) * stride_d + d) * stride_h + h) * stride_w + w;
}

__global__ void conv_transpose3d_kernel(
    const float* input, 
    const float* weight, 
    const float* bias, 
    float* output, 
    int batch_size, 
    int in_channels, 
    int out_channels, 
    int kernel_depth, 
    int kernel_height, 
    int kernel_width, 
    int stride_d, 
    int stride_h, 
    int stride_w, 
    int pad_d, 
    int pad_h, 
    int pad_w, 
    int in_depth, 
    int in_height, 
    int in_width, 
    int out_depth, 
    int out_height, 
    int out_width
) {
    // Calculate global thread index
    int total_output_elements = batch_size * out_channels * out_depth * out_height * out_width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_output_elements) return;

    // Decode linear index to 5D coordinates for output
    int w = idx % out_width;
    int h = (idx / out_width) % out_height;
    int d = (idx / (out_width * out_height)) % out_depth;
    int c_out = (idx / (out_width * out_height * out_depth)) % out_channels;
    int n = idx / (out_width * out_height * out_depth * out_channels);

    float sum = 0.0f;

    // Iterate over input channels and kernel dimensions
    // The receptive field in the input for output(n, c_out, d, h, w) is:
    // i_d ranges from max(0, d - pad_d + 1 - (kernel_depth-1)/2 * stride_d ? No, standard formula:
    // Input coordinate corresponding to output coordinate k with stride s and padding p:
    // i = (k - 1) * s - p + floor((k+1)*s/2) ... this is complex for transpose.
    // Standard ConvTranspose logic:
    // Output element at (d, h, w) receives contributions from Input elements at:
    // i_d in [d*stride_d - pad_d, d*stride_d - pad_d + kernel_depth - 1] intersected with valid input range [0, in_depth-1]
    
    int start_d = d * stride_d - pad_d;
    int start_h = h * stride_h - pad_h;
    int start_w = w * stride_w - pad_w;

    for (int k_d = 0; k_d < kernel_depth; ++k_d) {
        int i_d = start_d + k_d;
        if (i_d < 0 || i_d >= in_depth) continue;

        for (int k_h = 0; k_h < kernel_height; ++k_h) {
            int i_h = start_h + k_h;
            if (i_h < 0 || i_h >= in_height) continue;

            for (int k_w = 0; k_w < kernel_width; ++k_w) {
                int i_w = start_w + k_w;
                if (i_w < 0 || i_w >= in_width) continue;

                // Iterate over input channels
                for (int c_in = 0; c_in < in_channels; ++c_in) {
                    // Weight index: (out_channels, in_channels/groups, kernel_depth, kernel_height, kernel_width)
                    // Assuming groups=1 for this general implementation. If groups > 1, logic changes.
                    // For simplicity and given the prompt's generic nature, we assume standard conv without grouped complexity 
                    // or handle it if needed. The nn.ConvTranspose3d weight shape is (out_channels, in_channels/groups, kD, kH, kW).
                    
                    int w_idx = ((c_out * in_channels + c_in) * kernel_depth + k_d) * kernel_height + k_h) * kernel_width + k_w;
                    
                    // Input index: (batch_size, in_channels, in_depth, in_height, in_width)
                    int i_idx = ((n * in_channels + c_in) * in_depth + i_d) * in_height + i_h) * in_width + i_w;

                    sum += input[i_idx] * weight[w_idx];
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[c_out];
    }

    // Write to output
    int o_idx = ((n * out_channels + c_out) * out_depth + d) * out_height + h) * out_width + w;
    output[o_idx] = sum;
}

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias
) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_depth = input.size(2);
    auto in_height = input.size(3);
    auto in_width = input.size(4);

    auto out_channels = weight.size(0);
    auto kernel_depth = weight.size(2);
    auto kernel_height = weight.size(3);
    auto kernel_width = weight.size(4);

    // These parameters are typically passed or derived. 
    // Since we don't have the module object here, we assume standard stride=1, padding=0 for the kernel logic 
    // OR we need to pass them. The prompt example passes inputs. 
    // To make this robust, let's assume default stride=1, padding=0, output_padding=0 as per common simple cases 
    // or add arguments. Given the constraint "real code", I will add arguments to the function signature 
    // and update the Python wrapper to pass them.
    
    // However, looking at the prompt's `get_init_inputs`, it only provides in_channels, out_channels, kernel_size.
    // This implies we might need to hardcode stride=1, padding=0 or pass them from the model init.
    // Let's modify the CUDA function to accept these parameters explicitly.
    
    return torch::zeros({batch_size, out_channels, in_depth + 2*0 - kernel_depth + 1, in_height + 2*0 - kernel_height + 1, in_width + 2*0 - kernel_width + 1}); // Placeholder
}

// Redefine with explicit parameters for stride, padding, output_padding
torch::Tensor conv_transpose3d_cuda_full(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias,
    int stride_d,
    int stride_h,
    int stride_w,
    int pad_d,
    int pad_h,
    int pad_w,
    int output_pad_d,
    int output_pad_h,
    int output_pad_w
) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_depth = input.size(2);
    auto in_height = input.size(3);
    auto in_width = input.size(4);

    auto out_channels = weight.size(0);
    auto kernel_depth = weight.size(2);
    auto kernel_height = weight.size(3);
    auto kernel_width = weight.size(4);

    // Calculate output dimensions for ConvTranspose3d
    // O_d = (I_d - 1) * stride_d - 2*pad_d + kernel_depth + output_pad_d
    int out_depth = (in_depth - 1) * stride_d - 2 * pad_d + kernel_depth + output_pad_d;
    int out_height = (in_height - 1) * stride_h - 2 * pad_h + kernel_height + output_pad_h;
    int out_width = (in_width - 1) * stride_w - 2 * pad_w + kernel_width + output_pad_w;

    auto output = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width}, input.options());

    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.numel() > 0 ? bias.data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();

    int total_output_elements = batch_size * out_channels * out_depth * out_height * out_width;
    
    if (total_output_elements == 0) {
        return output;
    }

    const int block_size = 256;
    const int num_blocks = (total_output_elements + block_size - 1) / block_size;

    conv_transpose3d_kernel<<<num_blocks, block_size>>>(
        input_ptr, 
        weight_ptr, 
        bias_ptr, 
        output_ptr, 
        batch_size, 
        in_channels, 
        out_channels, 
        kernel_depth, 
        kernel_height, 
        kernel_width, 
        stride_d, 
        stride_h, 
        stride_w, 
        pad_d, 
        pad_h, 
        pad_w, 
        in_depth, 
        in_height, 
        in_width, 
        out_depth, 
        out_height, 
        out_width
    );

    return output;
}
"""

conv_transpose3d_cpp_source = (
    "torch::Tensor conv_transpose3d_cuda_full("
    "torch::Tensor input, "
    "torch::Tensor weight, "
    "torch::Tensor bias, "
    "int stride_d, int stride_h, int stride_w, "
    "int pad_d, int pad_h, int pad_w, "
    "int output_pad_d, int output_pad_h, int output_pad_w"
    ");"
)

# Compile the inline CUDA code
conv_transpose3d_module = load_inline(
    name="conv_transpose3d_cuda",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_cuda_full"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Optimized Transposed 3D Convolution using custom CUDA operator.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Store parameters for the forward pass
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and biases manually to match nn.ConvTranspose3d behavior
        # Weight shape: (out_channels, in_channels/groups, kD, kH, kW)
        # For groups=1, it's (out_channels, in_channels, kD, kH, kW)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming uniform initialization similar to PyTorch's default for ConvTranspose
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution using custom CUDA kernel.
        """
        # Handle groups > 1 is complex in a simple custom kernel without significant optimization gains 
        # over cuDNN for general cases. Assuming groups=1 for this optimized path as per typical "speedup" requests 
        # where standard ops are bottlenecked or to demonstrate custom logic. 
        # If groups != 1, we fall back to PyTorch's native implementation which is highly optimized via cuDNN.
        
        if self.groups != 1:
            return F.conv_transpose3d(
                x, 
                self.weight, 
                self.bias, 
                stride=self.stride, 
                padding=self.padding, 
                output_padding=self.output_padding, 
                groups=self.groups
            )

        # Extract dimensions
        batch_size = x.size(0)
        in_channels = x.size(1)
        in_depth = x.size(2)
        in_height = x.size(3)
        in_width = x.size(4)
        
        out_channels = self.out_channels
        kernel_size = self.kernel_size
        
        # Ensure tensors are contiguous and on CUDA
        x = x.contiguous()
        weight = self.weight.contiguous()
        bias = self.bias.contiguous() if self.bias is not None else torch.empty(0, device=x.device)

        # Call custom CUDA kernel
        output = conv_transpose3d_module.conv_transpose3d_cuda_full(
            x,
            weight,
            bias,
            self.stride, self.stride, self.stride,  # Assuming square stride
            self.padding, self.padding, self.padding,  # Assuming square padding
            self.output_padding, self.output_padding, self.output_padding  # Assuming square output_padding
        )
        
        return output

import math