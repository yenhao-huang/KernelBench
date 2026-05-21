import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source code for the fused kernel
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_groupnorm_swish_mul_swish_kernel(
    const float* input,
    float* output,
    const float* gamma,
    const float* beta,
    const float* multiply_weight,
    int batch_size,
    int num_groups,
    int channels_per_group,
    float eps
) {
    int group = blockIdx.x;
    int sample = blockIdx.y;
    int tid = threadIdx.x;
    int channel = group * channels_per_group + tid;
    int total_channels = num_groups * channels_per_group;

    float val = input[sample * total_channels + channel];

    // Warp reduce sum for mean
    float sum = val;
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    float mean = sum / channels_per_group;

    // Warp reduce sum for variance
    float diff = val - mean;
    float var_sum = diff * diff;
    for (int offset = 16; offset > 0; offset /= 2) {
        var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    }
    float var = var_sum / channels_per_group;

    // Normalize
    float inv_std = rsqrtf(var + eps);
    float normalized = (val - mean) * inv_std;
    float gn_out = normalized * gamma[channel] + beta[channel];

    // Swish activation
    float swish1 = gn_out * (1.0f / (1.0f + expf(-gn_out)));

    // Multiply with learned weight
    float mul = swish1 * multiply_weight[channel];

    // Second Swish activation
    float swish2 = mul * (1.0f / (1.0f + expf(-mul)));

    output[sample * total_channels + channel] = swish2;
}

torch::Tensor fused_groupnorm_swish_mul_swish_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor multiply_weight,
    int num_groups,
    float eps
) {
    int batch_size = input.size(0);
    int total_channels = input.size(1);
    int channels_per_group = total_channels / num_groups;

    auto output = torch::empty_like(input);

    dim3 grid(num_groups, batch_size);
    dim3 block(channels_per_group);

    fused_groupnorm_swish_mul_swish_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        multiply_weight.data_ptr<float>(),
        batch_size,
        num_groups,
        channels_per_group,
        eps
    );

    return output;
}
"""

cpp_source = "torch::Tensor fused_groupnorm_swish_mul_swish_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, torch::Tensor multiply_weight, int num_groups, float eps);"

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_groupnorm_swish_mul_swish",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_groupnorm_swish_mul_swish_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))
        self.fused_op = fused_op
        self.num_groups = num_groups
        self.eps = self.group_norm.eps

    def forward(self, x):
        x = self.gemm(x)
        x = self.fused_op.fused_groupnorm_swish_mul_swish_cuda(
            x,
            self.group_norm.weight,
            self.group_norm.bias,
            self.multiply_weight,
            self.num_groups,
            self.eps
        )
        return x