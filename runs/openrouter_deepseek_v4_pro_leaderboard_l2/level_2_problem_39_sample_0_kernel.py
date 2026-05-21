import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source code for fused scale + batch normalization
fused_scale_bn_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void reduce_scaled_sum_sq_kernel(const float* input, const float* scale, float* sum, float* sum_sq, int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C;
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        int col = i % C;
        float val = input[i] * scale[col];
        atomicAdd(&sum[col], val);
        atomicAdd(&sum_sq[col], val * val);
    }
}

__global__ void normalize_scaled_kernel(const float* input, const float* scale, const float* gamma, const float* beta,
                                        const float* sum, const float* sum_sq, float* output,
                                        int N, int C, float eps, float momentum,
                                        float* running_mean, float* running_var, bool training) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C;
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        int col = i % C;
        float mean = sum[col] / N;
        float var = sum_sq[col] / N - mean * mean;
        float inv_std = rsqrtf(var + eps);
        float val = input[i] * scale[col];
        output[i] = (val - mean) * inv_std * gamma[col] + beta[col];
    }
}

__global__ void update_running_stats_kernel(float* running_mean, float* running_var,
                                            const float* sum, const float* sum_sq,
                                            int N, int C, float momentum) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < C) {
        float mean = sum[col] / N;
        float var = sum_sq[col] / N - mean * mean;
        running_mean[col] = momentum * mean + (1.0f - momentum) * running_mean[col];
        running_var[col] = momentum * var + (1.0f - momentum) * running_var[col];
    }
}

__global__ void eval_normalize_scaled_kernel(const float* input, const float* scale, const float* gamma, const float* beta,
                                             const float* running_mean, const float* running_var,
                                             float* output, int N, int C, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C;
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        int col = i % C;
        float inv_std = rsqrtf(running_var[col] + eps);
        float val = input[i] * scale[col];
        output[i] = (val - running_mean[col]) * inv_std * gamma[col] + beta[col];
    }
}

torch::Tensor fused_scale_bn_cuda(torch::Tensor input, torch::Tensor scale, torch::Tensor gamma, torch::Tensor beta,
                                  torch::Tensor running_mean, torch::Tensor running_var,
                                  float eps, float momentum, bool training) {
    int N = input.size(0);
    int C = input.size(1);
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (N * C + block_size - 1) / block_size;

    if (training) {
        auto sum = torch::zeros({C}, input.options());
        auto sum_sq = torch::zeros({C}, input.options());

        reduce_scaled_sum_sq_kernel<<<num_blocks, block_size>>>(
            input.data_ptr<float>(), scale.data_ptr<float>(), sum.data_ptr<float>(), sum_sq.data_ptr<float>(), N, C);

        normalize_scaled_kernel<<<num_blocks, block_size>>>(
            input.data_ptr<float>(), scale.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
            sum.data_ptr<float>(), sum_sq.data_ptr<float>(), output.data_ptr<float>(),
            N, C, eps, momentum, running_mean.data_ptr<float>(), running_var.data_ptr<float>(), training);

        int stats_blocks = (C + block_size - 1) / block_size;
        update_running_stats_kernel<<<stats_blocks, block_size>>>(
            running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
            sum.data_ptr<float>(), sum_sq.data_ptr<float>(), N, C, momentum);
    } else {
        eval_normalize_scaled_kernel<<<num_blocks, block_size>>>(
            input.data_ptr<float>(), scale.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
            running_mean.data_ptr<float>(), running_var.data_ptr<float>(), output.data_ptr<float>(),
            N, C, eps);
    }

    return output;
}
"""

fused_scale_bn_cpp_source = "torch::Tensor fused_scale_bn_cuda(torch::Tensor input, torch::Tensor scale, torch::Tensor gamma, torch::Tensor beta, torch::Tensor running_mean, torch::Tensor running_var, float eps, float momentum, bool training);"

# Compile the inline CUDA code
fused_scale_bn = load_inline(
    name="fused_scale_bn",
    cpp_sources=fused_scale_bn_cpp_source,
    cuda_sources=fused_scale_bn_source,
    functions=["fused_scale_bn_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class FusedScaleBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(FusedScaleBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, input, scale):
        training = self.training
        return fused_scale_bn.fused_scale_bn_cuda(
            input, scale, self.weight, self.bias,
            self.running_mean, self.running_var,
            self.eps, self.momentum, training
        )

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.fused_scale_bn = FusedScaleBatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, x):
        x = self.gemm(x)
        x = self.fused_scale_bn(x, self.scale)
        return x