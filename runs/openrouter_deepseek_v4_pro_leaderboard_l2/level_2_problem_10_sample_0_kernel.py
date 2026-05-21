import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused transposed convolution + maxpool + hardtanh + mean + tanh
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_transpose_maxpool_hardtanh_mean_tanh_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int kernel_size,
    int stride,
    int padding,
    int maxpool_kernel_size,
    int maxpool_stride,
    float hardtanh_min,
    float hardtanh_max,
    int out_height,
    int out_width,
    int pooled_height,
    int pooled_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels;
    if (idx >= total_elements) return;

    int b = idx / out_channels;
    int oc = idx % out_channels;

    // Allocate shared memory for intermediate results
    extern __shared__ float shared_mem[];
    float* shared_output = shared_mem;
    float* shared_pooled = &shared_mem[out_height * out_width];
    
    // Initialize shared memory to zero
    for (int i = threadIdx.x; i < out_height * out_width; i += blockDim.x) {
        shared_output[i] = 0.0f;
    }
    __syncthreads();

    // Transposed convolution: compute output for this batch and output channel
    for (int ic = 0; ic < in_channels; ic++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                float w = weight[oc * in_channels * kernel_size * kernel_size + ic * kernel_size * kernel_size + kh * kernel_size + kw];
                for (int ih = 0; ih < in_height; ih++) {
                    for (int iw = 0; iw < in_width; iw++) {
                        int oh = ih * stride + kh - padding;
                        int ow = iw * stride + kw - padding;
                        if (oh >= 0 && oh < out_height && ow >= 0 && ow < out_width) {
                            float val = input[b * in_channels * in_height * in_width + ic * in_height * in_width + ih * in_width + iw];
                            atomicAdd(&shared_output[oh * out_width + ow], val * w);
                        }
                    }
                }
            }
        }
    }
    __syncthreads();

    // Add bias
    float bias_val = bias[oc];
    for (int i = threadIdx.x; i < out_height * out_width; i += blockDim.x) {
        shared_output[i] += bias_val;
    }
    __syncthreads();

    // Max pooling
    for (int ph = 0; ph < pooled_height; ph++) {
        for (int pw = 0; pw < pooled_width; pw++) {
            float max_val = -1e30f;
            for (int mh = 0; mh < maxpool_kernel_size; mh++) {
                for (int mw = 0; mw < maxpool_kernel_size; mw++) {
                    int oh = ph * maxpool_stride + mh;
                    int ow = pw * maxpool_stride + mw;
                    if (oh < out_height && ow < out_width) {
                        float val = shared_output[oh * out_width + ow];
                        if (val > max_val) max_val = val;
                    }
                }
            }
            shared_pooled[ph * pooled_width + pw] = max_val;
        }
    }
    __syncthreads();

    // Hardtanh activation
    for (int i = threadIdx.x; i < pooled_height * pooled_width; i += blockDim.x) {
        float val = shared_pooled[i];
        if (val < hardtanh_min) val = hardtanh_min;
        else if (val > hardtanh_max) val = hardtanh_max;
        shared_pooled[i] = val;
    }
    __syncthreads();

    // Mean over spatial dimensions
    float sum = 0.0f;
    for (int i = threadIdx.x; i < pooled_height * pooled_width; i += blockDim.x) {
        sum += shared_pooled[i];
    }
    // Reduce sum within block
    __shared__ float block_sum[256];
    block_sum[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    float mean_val = block_sum[0] / (pooled_height * pooled_width);

    // Tanh activation
    float tanh_val = tanhf(mean_val);
    
    // Write output
    if (threadIdx.x == 0) {
        output[b * out_channels + oc] = tanh_val;
    }
}

torch::Tensor fused_ops_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int stride,
    int padding,
    int maxpool_kernel_size,
    int maxpool_stride,
    float hardtanh_min,
    float hardtanh_max
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    int out_channels = weight.size(0);
    
    int out_height = (in_height - 1) * stride - 2 * padding + kernel_size;
    int out_width = (in_width - 1) * stride - 2 * padding + kernel_size;
    int pooled_height = (out_height - maxpool_kernel_size) / maxpool_stride + 1;
    int pooled_width = (out_width - maxpool_kernel_size) / maxpool_stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, 1, 1}, input.options());
    
    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels + block_size - 1) / block_size;
    size_t shared_mem_size = (out_height * out_width + pooled_height * pooled_width) * sizeof(float);
    
    fused_conv_transpose_maxpool_hardtanh_mean_tanh_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        kernel_size,
        stride,
        padding,
        maxpool_kernel_size,
        maxpool_stride,
        hardtanh_min,
        hardtanh_max,
        out_height,
        out_width,
        pooled_height,
        pooled_width
    );
    
    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_ops_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int stride,
    int padding,
    int maxpool_kernel_size,
    int maxpool_stride,
    float hardtanh_min,
    float hardtanh_max
);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.maxpool_kernel_size = maxpool_kernel_size
        self.maxpool_stride = maxpool_stride
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self.fused_ops = fused_ops

    def forward(self, x):
        weight = self.conv_transpose.weight
        bias = self.conv_transpose.bias
        return self.fused_ops.fused_ops_cuda(
            x,
            weight,
            bias,
            self.conv_transpose.kernel_size[0],
            self.conv_transpose.stride[0],
            self.conv_transpose.padding[0],
            self.maxpool_kernel_size,
            self.maxpool_stride,
            self.hardtanh_min,
            self.hardtanh_max
        )