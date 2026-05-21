import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for ConvTranspose3d + BatchNorm3d + Subtract Mean
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Kernel for ConvTranspose3d forward pass (simplified, assumes stride=2, padding=1, kernel=3)
__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_d * out_h * out_w;
    if (idx >= total_elements) return;

    // Decompose linear index
    int w = idx % out_w;
    int h = (idx / out_w) % out_h;
    int d = (idx / (out_w * out_h)) % out_d;
    int oc = (idx / (out_w * out_h * out_d)) % out_channels;
    int n = idx / (out_w * out_h * out_d * out_channels);

    float value = bias != nullptr ? bias[oc] : 0.0f;

    // Iterate over input positions and kernel
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kd = 0; kd < kernel_d; ++kd) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int in_d_idx = d + padding_d - kd;
                    int in_h_idx = h + padding_h - kh;
                    int in_w_idx = w + padding_w - kw;

                    if (in_d_idx >= 0 && in_d_idx < in_d * stride_d && in_d_idx % stride_d == 0 &&
                        in_h_idx >= 0 && in_h_idx < in_h * stride_h && in_h_idx % stride_h == 0 &&
                        in_w_idx >= 0 && in_w_idx < in_w * stride_w && in_w_idx % stride_w == 0) {
                        int in_d_pos = in_d_idx / stride_d;
                        int in_h_pos = in_h_idx / stride_h;
                        int in_w_pos = in_w_idx / stride_w;
                        if (in_d_pos < in_d && in_h_pos < in_h && in_w_pos < in_w) {
                            float input_val = input[((n * in_channels + ic) * in_d + in_d_pos) * in_h * in_w + in_h_pos * in_w + in_w_pos];
                            float weight_val = weight[((oc * in_channels + ic) * kernel_d + kd) * kernel_h * kernel_w + kh * kernel_w + kw];
                            value += input_val * weight_val;
                        }
                    }
                }
            }
        }
    }

    output[idx] = value;
}

// Kernel for BatchNorm3d + Subtract Mean fusion
__global__ void batchnorm_subtract_mean_kernel(
    float* __restrict__ data,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int batch_size, int channels, int spatial_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * spatial_size;
    if (idx >= total_elements) return;

    int spatial_idx = idx % spatial_size;
    int c = (idx / spatial_size) % channels;
    int n = idx / (spatial_size * channels);

    // BatchNorm
    float x_hat = (data[idx] - running_mean[c]) / sqrtf(running_var[c] + eps);
    float y = gamma[c] * x_hat + beta[c];
    data[idx] = y;

    // Compute mean per channel per batch using shared memory
    __shared__ float shared_sum[256];
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Each block handles one channel of one batch
    if (spatial_idx == 0) {
        shared_sum[tid] = 0.0f;
    }
    __syncthreads();

    // Accumulate sum for this channel
    float local_sum = 0.0f;
    for (int i = tid; i < spatial_size; i += block_size) {
        int global_idx = (n * channels + c) * spatial_size + i;
        local_sum += data[global_idx];
    }
    atomicAdd(&shared_sum[0], local_sum);
    __syncthreads();

    float mean = shared_sum[0] / spatial_size;

    // Subtract mean
    data[idx] -= mean;
}

torch::Tensor fused_conv_transpose_bn_sub_mean_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps,
    int stride, int padding
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_d = input.size(2);
    int in_h = input.size(3);
    int in_w = input.size(4);
    int out_channels = weight.size(0);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);

    int out_d = (in_d - 1) * stride - 2 * padding + kernel_d;
    int out_h = (in_h - 1) * stride - 2 * padding + kernel_h;
    int out_w = (in_w - 1) * stride - 2 * padding + kernel_w;

    auto output = torch::zeros({batch_size, out_channels, out_d, out_h, out_w}, input.options());

    int total_conv_elements = batch_size * out_channels * out_d * out_h * out_w;
    const int block_size = 256;
    const int num_blocks = (total_conv_elements + block_size - 1) / block_size;

    conv_transpose3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_d, kernel_h, kernel_w,
        stride, stride, stride,
        padding, padding, padding
    );

    int spatial_size = out_d * out_h * out_w;
    int total_bn_elements = batch_size * out_channels * spatial_size;
    const int bn_blocks = (total_bn_elements + block_size - 1) / block_size;

    batchnorm_subtract_mean_kernel<<<bn_blocks, block_size>>>(
        output.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        eps,
        batch_size, out_channels, spatial_size
    );

    return output;
}
"""

cpp_source = """
torch::Tensor fused_conv_transpose_bn_sub_mean_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps,
    int stride, int padding
);
"""

fused_op = load_inline(
    name="fused_conv_transpose_bn_sub_mean",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_conv_transpose_bn_sub_mean_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.fused_op = fused_op
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return self.fused_op.fused_conv_transpose_bn_sub_mean_cuda(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.eps,
            self.stride,
            self.padding
        )