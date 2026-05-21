import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused Conv3d + GroupNorm + Mean reduction
fused_conv_gn_mean_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Kernel for fused Conv3d + GroupNorm + Mean reduction
// This kernel performs convolution, group normalization, and computes the mean
// in a single pass to reduce memory bandwidth and improve performance.
__global__ void fused_conv_gn_mean_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ gn_weight,
    const float* __restrict__ gn_bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int D,
    int H,
    int W,
    int kernel_size,
    int num_groups,
    float eps
) {
    // Each block handles one output element (batch, oc, d, h, w)
    int b = blockIdx.z;
    int oc = blockIdx.y;
    int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int D_out = D - kernel_size + 1;
    int H_out = H - kernel_size + 1;
    int W_out = W - kernel_size + 1;
    int total_spatial = D_out * H_out * W_out;
    
    if (spatial_idx >= total_spatial) return;
    
    int d_out = spatial_idx / (H_out * W_out);
    int rem = spatial_idx % (H_out * W_out);
    int h_out = rem / W_out;
    int w_out = rem % W_out;
    
    // Compute convolution for this output element
    float conv_val = bias[oc];
    for (int ic = 0; ic < in_channels; ic++) {
        for (int kd = 0; kd < kernel_size; kd++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int d_in = d_out + kd;
                    int h_in = h_out + kh;
                    int w_in = w_out + kw;
                    int input_idx = ((b * in_channels + ic) * D + d_in) * H + h_in;
                    input_idx = input_idx * W + w_in;
                    int weight_idx = (((oc * in_channels + ic) * kernel_size + kd) * kernel_size + kh) * kernel_size + kw;
                    conv_val += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Store convolution result in shared memory for group norm
    extern __shared__ float shared_vals[];
    float* conv_vals = shared_vals;
    float* group_stats = &shared_vals[total_spatial];
    
    conv_vals[spatial_idx] = conv_val;
    __syncthreads();
    
    // Compute group normalization
    int group = oc / (out_channels / num_groups);
    int group_size = out_channels / num_groups;
    int local_oc = oc % group_size;
    
    // Compute mean and variance for this group across spatial dimensions
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        float sq_sum = 0.0f;
        for (int i = 0; i < total_spatial; i++) {
            float val = conv_vals[i];
            sum += val;
            sq_sum += val * val;
        }
        float mean = sum / total_spatial;
        float var = sq_sum / total_spatial - mean * mean;
        group_stats[0] = mean;
        group_stats[1] = var;
    }
    __syncthreads();
    
    float mean = group_stats[0];
    float var = group_stats[1];
    float inv_std = rsqrtf(var + eps);
    
    // Normalize
    float normalized = (conv_val - mean) * inv_std;
    float gn_result = normalized * gn_weight[oc] + gn_bias[oc];
    
    // Atomic add for mean reduction across all dimensions except batch
    // We use atomicAdd to accumulate the sum for each batch element
    atomicAdd(&output[b], gn_result / (out_channels * D_out * H_out * W_out));
}

torch::Tensor fused_conv_gn_mean_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    int kernel_size,
    int num_groups,
    float eps
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);
    int out_channels = weight.size(0);
    
    int D_out = D - kernel_size + 1;
    int H_out = H - kernel_size + 1;
    int W_out = W - kernel_size + 1;
    int total_spatial = D_out * H_out * W_out;
    
    auto output = torch::zeros({batch_size}, torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));
    
    dim3 grid(total_spatial, out_channels, batch_size);
    dim3 block(256);
    
    int shared_mem_size = total_spatial * sizeof(float) + 2 * sizeof(float);
    
    fused_conv_gn_mean_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        gn_weight.data_ptr<float>(),
        gn_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        D,
        H,
        W,
        kernel_size,
        num_groups,
        eps
    );
    
    return output;
}
"""

fused_conv_gn_mean_cpp_source = (
    "torch::Tensor fused_conv_gn_mean_cuda("
    "torch::Tensor input, "
    "torch::Tensor weight, "
    "torch::Tensor bias, "
    "torch::Tensor gn_weight, "
    "torch::Tensor gn_bias, "
    "int kernel_size, "
    "int num_groups, "
    "float eps"
    ");"
)

# Compile the inline CUDA code
fused_conv_gn_mean = load_inline(
    name="fused_conv_gn_mean",
    cpp_sources=fused_conv_gn_mean_cpp_source,
    cuda_sources=fused_conv_gn_mean_source,
    functions=["fused_conv_gn_mean_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.fused_conv_gn_mean = fused_conv_gn_mean
        self.kernel_size = kernel_size
        self.num_groups = num_groups

    def forward(self, x):
        # Use fused kernel that combines Conv3d, GroupNorm, and Mean reduction
        return self.fused_conv_gn_mean.fused_conv_gn_mean_cuda(
            x,
            self.conv.weight,
            self.conv.bias,
            self.group_norm.weight,
            self.group_norm.bias,
            self.kernel_size,
            self.num_groups,
            self.group_norm.eps
        )