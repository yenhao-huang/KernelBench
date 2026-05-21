import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for Conv3d + Softmax + MaxPool3d fusion
conv_softmax_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Kernel for Conv3d + Softmax + MaxPool3d fusion
// This kernel performs convolution, applies softmax along channel dimension,
// and then applies two max pooling operations in a single pass
__global__ void conv_softmax_pool_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int depth,
    int height,
    int width,
    int kernel_size,
    int pool_kernel_size,
    int out_depth,
    int out_height,
    int out_width,
    int conv_out_depth,
    int conv_out_height,
    int conv_out_width,
    int pool1_out_depth,
    int pool1_out_height,
    int pool1_out_width
) {
    // Each thread handles one output element (batch, channel, depth, height, width)
    int b = blockIdx.z;
    int oc = blockIdx.y;
    int od = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (b >= batch_size || oc >= out_channels || od >= out_depth) return;
    
    int oh_start = threadIdx.y;
    int ow_start = threadIdx.z;
    
    // We'll process one spatial position per thread
    for (int oh = oh_start; oh < out_height; oh += blockDim.y) {
        for (int ow = ow_start; ow < out_width; ow += blockDim.z) {
            // Map output position back to input region for convolution
            int conv_d_start = od * pool_kernel_size * pool_kernel_size;
            int conv_h_start = oh * pool_kernel_size * pool_kernel_size;
            int conv_w_start = ow * pool_kernel_size * pool_kernel_size;
            
            // Find max over the pooling region (two pooling operations combined)
            float max_val = -1e38f;
            
            for (int pd1 = 0; pd1 < pool_kernel_size; pd1++) {
                for (int ph1 = 0; ph1 < pool_kernel_size; ph1++) {
                    for (int pw1 = 0; pw1 < pool_kernel_size; pw1++) {
                        int d1 = conv_d_start + pd1;
                        int h1 = conv_h_start + ph1;
                        int w1 = conv_w_start + pw1;
                        
                        if (d1 >= conv_out_depth || h1 >= conv_out_height || w1 >= conv_out_width) continue;
                        
                        for (int pd2 = 0; pd2 < pool_kernel_size; pd2++) {
                            for (int ph2 = 0; ph2 < pool_kernel_size; ph2++) {
                                for (int pw2 = 0; pw2 < pool_kernel_size; pw2++) {
                                    int d2 = d1 * pool_kernel_size + pd2;
                                    int h2 = h1 * pool_kernel_size + ph2;
                                    int w2 = w1 * pool_kernel_size + pw2;
                                    
                                    if (d2 >= conv_out_depth || h2 >= conv_out_height || w2 >= conv_out_width) continue;
                                    
                                    // Compute convolution for this position
                                    float conv_val = bias[oc];
                                    for (int ic = 0; ic < in_channels; ic++) {
                                        for (int kd = 0; kd < kernel_size; kd++) {
                                            for (int kh = 0; kh < kernel_size; kh++) {
                                                for (int kw = 0; kw < kernel_size; kw++) {
                                                    int in_d = d2 + kd;
                                                    int in_h = h2 + kh;
                                                    int in_w = w2 + kw;
                                                    
                                                    if (in_d < depth && in_h < height && in_w < width) {
                                                        conv_val += input[((b * in_channels + ic) * depth + in_d) * height * width + in_h * width + in_w] *
                                                                   weight[((oc * in_channels + ic) * kernel_size + kd) * kernel_size * kernel_size + kh * kernel_size + kw];
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    
                                    // Apply softmax later, just store conv value for now
                                    if (conv_val > max_val) max_val = conv_val;
                                }
                            }
                        }
                    }
                }
            }
            
            // Now compute softmax for this spatial position across channels
            // We need to recompute conv values for all channels at this position
            // This is inefficient but demonstrates the fusion concept
            // In practice, you'd use shared memory or tiling
            
            float sum_exp = 0.0f;
            float exp_vals[32]; // Assume max 32 channels for simplicity
            float conv_vals[32];
            
            for (int c = 0; c < out_channels; c++) {
                float conv_val = bias[c];
                for (int ic = 0; ic < in_channels; ic++) {
                    for (int kd = 0; kd < kernel_size; kd++) {
                        for (int kh = 0; kh < kernel_size; kh++) {
                            for (int kw = 0; kw < kernel_size; kw++) {
                                int in_d = conv_d_start * pool_kernel_size * pool_kernel_size + kd;
                                int in_h = conv_h_start * pool_kernel_size * pool_kernel_size + kh;
                                int in_w = conv_w_start * pool_kernel_size * pool_kernel_size + kw;
                                
                                if (in_d < depth && in_h < height && in_w < width) {
                                    conv_val += input[((b * in_channels + ic) * depth + in_d) * height * width + in_h * width + in_w] *
                                               weight[((c * in_channels + ic) * kernel_size + kd) * kernel_size * kernel_size + kh * kernel_size + kw];
                                }
                            }
                        }
                    }
                }
                conv_vals[c] = conv_val;
                exp_vals[c] = expf(conv_val - max_val);
                sum_exp += exp_vals[c];
            }
            
            // Write softmax output for this channel
            float softmax_val = exp_vals[oc] / sum_exp;
            output[((b * out_channels + oc) * out_depth + od) * out_height * out_width + oh * out_width + ow] = softmax_val;
        }
    }
}

torch::Tensor conv_softmax_pool_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int pool_kernel_size
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);
    int out_channels = weight.size(0);
    
    int conv_out_depth = depth - kernel_size + 1;
    int conv_out_height = height - kernel_size + 1;
    int conv_out_width = width - kernel_size + 1;
    
    int pool1_out_depth = conv_out_depth / pool_kernel_size;
    int pool1_out_height = conv_out_height / pool_kernel_size;
    int pool1_out_width = conv_out_width / pool_kernel_size;
    
    int out_depth = pool1_out_depth / pool_kernel_size;
    int out_height = pool1_out_height / pool_kernel_size;
    int out_width = pool1_out_width / pool_kernel_size;
    
    auto output = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width}, input.options());
    
    dim3 block(16, 4, 4);
    dim3 grid(
        (out_depth + block.x - 1) / block.x,
        out_channels,
        batch_size
    );
    
    conv_softmax_pool_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        depth,
        height,
        width,
        kernel_size,
        pool_kernel_size,
        out_depth,
        out_height,
        out_width,
        conv_out_depth,
        conv_out_height,
        conv_out_width,
        pool1_out_depth,
        pool1_out_height,
        pool1_out_width
    );
    
    return output;
}
"""

conv_softmax_pool_cpp_source = """
torch::Tensor conv_softmax_pool_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int pool_kernel_size
);
"""

# Compile the inline CUDA code
conv_softmax_pool = load_inline(
    name="conv_softmax_pool",
    cpp_sources=conv_softmax_pool_cpp_source,
    cuda_sources=conv_softmax_pool_source,
    functions=["conv_softmax_pool_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.pool_kernel_size = pool_kernel_size
        self.kernel_size = kernel_size
        self.conv_softmax_pool = conv_softmax_pool

    def forward(self, x):
        return self.conv_softmax_pool.conv_softmax_pool_cuda(
            x,
            self.conv.weight,
            self.conv.bias,
            self.kernel_size,
            self.pool_kernel_size
        )