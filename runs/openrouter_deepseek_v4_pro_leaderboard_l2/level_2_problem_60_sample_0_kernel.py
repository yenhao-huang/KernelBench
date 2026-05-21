import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for fused Swish + GroupNorm + HardSwish
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void compute_mean_var_swish_kernel(
    const float* __restrict__ input,
    float* __restrict__ mean,
    float* __restrict__ var,
    int N, int C, int D, int H, int W, int groups)
{
    int spatial_size = D * H * W;
    int channels_per_group = C / groups;
    int total_elements_per_group = channels_per_group * spatial_size;

    int n = blockIdx.x / groups;
    int g = blockIdx.x % groups;

    extern __shared__ float shared[];
    float* sum_buf = shared;
    float* sq_sum_buf = shared + blockDim.x;

    int tid = threadIdx.x;
    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;

    int base_channel = g * channels_per_group;
    int input_offset = n * C * spatial_size;

    for (int idx = tid; idx < total_elements_per_group; idx += blockDim.x) {
        int c = idx / spatial_size;
        int s = idx % spatial_size;
        int channel = base_channel + c;
        float val = input[input_offset + channel * spatial_size + s];
        // Swish: sigmoid(x) * x
        float sigmoid_val = 1.0f / (1.0f + expf(-val));
        float swish_val = sigmoid_val * val;
        local_sum += swish_val;
        local_sq_sum += swish_val * swish_val;
    }

    sum_buf[tid] = local_sum;
    sq_sum_buf[tid] = local_sq_sum;
    __syncthreads();

    // Reduction within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_buf[tid] += sum_buf[tid + stride];
            sq_sum_buf[tid] += sq_sum_buf[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float total_sum = sum_buf[0];
        float total_sq_sum = sq_sum_buf[0];
        float mean_val = total_sum / total_elements_per_group;
        float var_val = total_sq_sum / total_elements_per_group - mean_val * mean_val;
        int out_idx = n * groups + g;
        mean[out_idx] = mean_val;
        var[out_idx] = var_val;
    }
}

__global__ void normalize_affine_hardswish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int D, int H, int W, int groups, float eps)
{
    int spatial_size = D * H * W;
    int channels_per_group = C / groups;
    int total_elements = N * C * spatial_size;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_elements) return;

    int n = idx / (C * spatial_size);
    int rem = idx % (C * spatial_size);
    int c = rem / spatial_size;
    int s = rem % spatial_size;

    int g = c / channels_per_group;
    int mean_var_idx = n * groups + g;
    float mean_val = mean[mean_var_idx];
    float var_val = var[mean_var_idx];
    float inv_std = rsqrtf(var_val + eps);

    // Swish
    float val = input[idx];
    float sigmoid_val = 1.0f / (1.0f + expf(-val));
    float swish_val = sigmoid_val * val;

    // Normalize
    float norm_val = (swish_val - mean_val) * inv_std;

    // Affine
    float gamma = weight[c];
    float beta = (bias != nullptr) ? bias[c] : 0.0f;
    float affine_val = gamma * norm_val + beta;

    // HardSwish
    float hardswish_val = affine_val;
    float relu6_arg = affine_val + 3.0f;
    relu6_arg = fminf(fmaxf(relu6_arg, 0.0f), 6.0f);
    hardswish_val = affine_val * relu6_arg * (1.0f / 6.0f);

    output[idx] = hardswish_val;
}

torch::Tensor swish_group_norm_hardswish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int groups,
    float eps)
{
    const auto N = input.size(0);
    const auto C = input.size(1);
    const auto D = input.size(2);
    const auto H = input.size(3);
    const auto W = input.size(4);

    auto output = torch::empty_like(input);
    auto mean = torch::empty({N, groups}, input.options());
    auto var = torch::empty({N, groups}, input.options());

    const int block_size = 256;
    const int grid_size_mean = N * groups;
    const int shared_mem_size = 2 * block_size * sizeof(float);

    compute_mean_var_swish_kernel<<<grid_size_mean, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        N, C, D, H, W, groups);

    const int total_elements = N * C * D * H * W;
    const int grid_size_norm = (total_elements + block_size - 1) / block_size;

    normalize_affine_hardswish_kernel<<<grid_size_norm, block_size>>>(
        input.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C, D, H, W, groups, eps);

    return output;
}
"""

fused_op_cpp_source = """
torch::Tensor swish_group_norm_hardswish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int groups,
    float eps);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_swish_group_norm_hardswish",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["swish_group_norm_hardswish_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )
        self.groups = groups
        self.eps = eps
        # GroupNorm parameters (weight and bias) are registered as buffers/parameters
        self.weight = nn.Parameter(torch.ones(out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv_transpose(x)
        # Fused Swish + GroupNorm + HardSwish
        x = self.fused_op.swish_group_norm_hardswish_cuda(
            x, self.weight, self.bias, self.groups, self.eps
        )
        return x