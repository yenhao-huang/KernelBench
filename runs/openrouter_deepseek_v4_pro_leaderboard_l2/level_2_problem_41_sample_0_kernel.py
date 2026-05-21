import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA source: fusion of BatchNorm (reduce) and BatchNorm+GELU+ReLU activation
fused_bn_gelu_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// This kernel computes per-channel mean and (biased) variance over the batch dimension.
// Input: shape (N, C) row-major contiguous.
// Output: mean (C), var (C)
__global__ void batch_norm_reduce_kernel(
    const float* __restrict__ input,
    float* __restrict__ mean_out,
    float* __restrict__ var_out,
    int N, int C) {

    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;

    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (int i = 0; i < N; ++i) {
        float val = input[i * C + c];
        sum += val;
        sum_sq += val * val;
    }
    float mean = sum / static_cast<float>(N);
    float var = sum_sq / static_cast<float>(N) - mean * mean;
    mean_out[c] = mean;
    var_out[c] = var;
}

// GELU approximation (tanh version)
__device__ float gelu_approx(float x) {
    const float alpha = 0.7978845608028654f; // sqrtf(2.0f / M_PI)
    float x3 = x * x * x;
    float tanh_arg = alpha * (x + 0.044715f * x3);
    return 0.5f * x * (1.0f + tanhf(tanh_arg));
}

// Fused kernel: BatchNorm + GELU + ReLU
// Pre‑computed mean and var are passed (could be batch statistics or running estimates)
__global__ void fused_bn_gelu_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float eps,
    float* __restrict__ output,
    int N, int C) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C;
    if (idx >= total_elements) return;

    int n = idx / C;
    int c = idx % C;

    float val = (input[idx] - mean[c]) * rsqrtf(var[c] + eps);
    val = gamma[c] * val + beta[c];

    // GELU
    float gelu_val = gelu_approx(val);
    // ReLU
    gelu_val = fmaxf(gelu_val, 0.0f);
    output[idx] = gelu_val;
}

// Wrapper for the reduce step
torch::Tensor batch_norm_reduce_cuda(torch::Tensor input) {
    AT_ASSERTM(input.dim() == 2, "Expected 2D input");
    const int N = input.size(0);
    const int C = input.size(1);
    auto mean = torch::empty({C}, input.options());
    auto var = torch::empty({C}, input.options());

    const int threads = 256;
    const int blocks = (C + threads - 1) / threads;

    batch_norm_reduce_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        N, C);

    return std::make_tuple(mean, var);
}

// Wrapper for the fused activation kernel
torch::Tensor fused_bn_gelu_relu_cuda(
    torch::Tensor input,
    torch::Tensor mean,
    torch::Tensor var,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps) {

    AT_ASSERTM(input.dim() == 2, "Expected 2D input");
    const int N = input.size(0);
    const int C = input.size(1);
    auto output = torch::empty_like(input);

    const int threads = 256;
    const int total_elements = N * C;
    const int blocks = (total_elements + threads - 1) / threads;

    fused_bn_gelu_relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        eps,
        output.data_ptr<float>(),
        N, C);

    return output;
}
"""

fused_bn_gelu_relu_cpp_source = """
#include <torch/extension.h>
torch::Tensor batch_norm_reduce_cuda(torch::Tensor input);
torch::Tensor fused_bn_gelu_relu_cuda(torch::Tensor input, torch::Tensor mean, torch::Tensor var, torch::Tensor gamma, torch::Tensor beta, float eps);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_bn_gelu_relu",
    cpp_sources=fused_bn_gelu_relu_cpp_source,
    cuda_sources=fused_bn_gelu_relu_source,
    functions=["batch_norm_reduce_cuda", "fused_bn_gelu_relu_cuda"],
    verbose=False,
    extra_cflags=[],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    """
    Optimized model that replaces the sequence BatchNorm -> GELU -> ReLU
    with a single fused custom CUDA kernel. The linear layer remains as cuBLAS.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)
        # Pre-compiled custom operators
        self.batch_norm_reduce = fused_ops.batch_norm_reduce_cuda
        self.fused_bn_gelu_relu = fused_ops.fused_bn_gelu_relu_cuda

    def forward(self, x):
        # 1. GEMM (cuBLAS)
        x = self.gemm(x)                  # shape: (N, C)

        # 2. Fused BatchNorm + GELU + ReLU
        if self.training:
            # Compute batch statistics (mean, var)
            mean, var = self.batch_norm_reduce(x)
            # Update running statistics (mimics PyTorch BatchNorm1d behaviour)
            m = self.batch_norm.momentum
            self.batch_norm.running_mean = m * self.batch_norm.running_mean + (1 - m) * mean
            self.batch_norm.running_var  = m * self.batch_norm.running_var  + (1 - m) * var
            # Use batch statistics for normalization
            out = self.fused_bn_gelu_relu(
                x, mean, var,
                self.batch_norm.weight,
                self.batch_norm.bias,
                self.batch_norm.eps)
        else:
            # Evaluation mode: use stored running statistics
            out = self.fused_bn_gelu_relu(
                x,
                self.batch_norm.running_mean,
                self.batch_norm.running_var,
                self.batch_norm.weight,
                self.batch_norm.bias,
                self.batch_norm.eps)
        return out