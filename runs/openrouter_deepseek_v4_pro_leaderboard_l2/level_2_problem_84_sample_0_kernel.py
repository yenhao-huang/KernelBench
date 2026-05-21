import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused Gemm + BatchNorm + Scale + Softmax
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Fused kernel: Gemm (matrix multiply) + BatchNorm + Scale + Softmax
// Input: x (batch_size, in_features)
// Weight: weight (out_features, in_features)
// Bias: bias (out_features)
// BN: running_mean, running_var, weight_bn, bias_bn, eps
// Scale: scale (scalar or broadcastable)
// Output: out (batch_size, out_features)

__global__ void fused_gemm_bn_scale_softmax_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    const float* __restrict__ weight_bn,
    const float* __restrict__ bias_bn,
    const float* __restrict__ scale,
    float* __restrict__ out,
    int batch_size,
    int in_features,
    int out_features,
    float eps
) {
    // Each block handles one row of the output (one sample in batch)
    int row = blockIdx.x;
    if (row >= batch_size) return;

    // Shared memory for partial sums during Gemm
    extern __shared__ float shared_mem[];
    float* partial_sums = shared_mem; // size: out_features
    float* max_vals = &shared_mem[out_features]; // for softmax max reduction
    float* sum_exp = &shared_mem[out_features + blockDim.x]; // for softmax sum

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // Initialize partial sums for this row
    for (int i = tid; i < out_features; i += num_threads) {
        partial_sums[i] = 0.0f;
    }
    __syncthreads();

    // Gemm: compute dot product for each output feature
    // Each thread computes partial sums for multiple output features
    for (int i = tid; i < out_features; i += num_threads) {
        float sum = 0.0f;
        const float* x_row = x + row * in_features;
        const float* w_row = weight + i * in_features;
        for (int j = 0; j < in_features; j++) {
            sum += x_row[j] * w_row[j];
        }
        sum += bias[i];
        partial_sums[i] = sum;
    }
    __syncthreads();

    // BatchNorm: normalize using running stats
    for (int i = tid; i < out_features; i += num_threads) {
        float val = partial_sums[i];
        float mean = running_mean[i];
        float var = running_var[i];
        float inv_std = rsqrtf(var + eps);
        float normalized = (val - mean) * inv_std;
        // Apply BN weight and bias
        partial_sums[i] = weight_bn[i] * normalized + bias_bn[i];
    }
    __syncthreads();

    // Scale
    float scale_val = scale[0]; // assuming scale is scalar
    for (int i = tid; i < out_features; i += num_threads) {
        partial_sums[i] *= scale_val;
    }
    __syncthreads();

    // Softmax: compute max for numerical stability
    // First, each thread finds local max among its assigned elements
    float local_max = -INFINITY;
    for (int i = tid; i < out_features; i += num_threads) {
        float val = partial_sums[i];
        if (val > local_max) local_max = val;
    }
    // Store local max in shared memory for reduction
    max_vals[tid] = local_max;
    __syncthreads();

    // Parallel reduction to find global max
    for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (max_vals[tid + stride] > max_vals[tid]) {
                max_vals[tid] = max_vals[tid + stride];
            }
        }
        __syncthreads();
    }
    float global_max = max_vals[0];
    __syncthreads();

    // Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (int i = tid; i < out_features; i += num_threads) {
        float val = expf(partial_sums[i] - global_max);
        partial_sums[i] = val; // reuse partial_sums to store exp values
        local_sum += val;
    }
    sum_exp[tid] = local_sum;
    __syncthreads();

    // Reduce sum
    for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_exp[tid] += sum_exp[tid + stride];
        }
        __syncthreads();
    }
    float global_sum = sum_exp[0];
    __syncthreads();

    // Normalize and write output
    for (int i = tid; i < out_features; i += num_threads) {
        out[row * out_features + i] = partial_sums[i] / global_sum;
    }
}

torch::Tensor fused_gemm_bn_scale_softmax_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight_bn,
    torch::Tensor bias_bn,
    torch::Tensor scale,
    float eps
) {
    int batch_size = x.size(0);
    int in_features = x.size(1);
    int out_features = weight.size(0);

    auto out = torch::empty({batch_size, out_features}, x.options());

    int threads = 256;
    int blocks = batch_size;

    // Shared memory: out_features floats for partial sums + blockDim.x floats for max + blockDim.x floats for sum
    // Actually we need out_features for partial sums, and then max_vals and sum_exp arrays of size threads.
    // But we can allocate enough shared memory dynamically.
    size_t shared_mem_size = (out_features + 2 * threads) * sizeof(float);

    fused_gemm_bn_scale_softmax_kernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        weight_bn.data_ptr<float>(),
        bias_bn.data_ptr<float>(),
        scale.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        eps
    );

    return out;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_gemm_bn_scale_softmax_cuda("
    "    torch::Tensor x,"
    "    torch::Tensor weight,"
    "    torch::Tensor bias,"
    "    torch::Tensor running_mean,"
    "    torch::Tensor running_var,"
    "    torch::Tensor weight_bn,"
    "    torch::Tensor bias_bn,"
    "    torch::Tensor scale,"
    "    float eps"
    ");"
)

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_gemm_bn_scale_softmax_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.bn_eps = bn_eps
        self.fused_ops = fused_ops

    def forward(self, x):
        # Use fused CUDA kernel for training/eval? 
        # For simplicity, we assume eval mode (use running stats) as the original model does not specify mode.
        # In training mode, BatchNorm uses batch stats, which would require a different kernel.
        # Here we implement the forward pass using the fused kernel that uses running stats (eval mode).
        # To match original behavior, we need to handle both train and eval. 
        # For simplicity, we assume eval mode. If training is needed, we could fallback to original ops.
        if self.training:
            # Fallback to original implementation for training (since BN uses batch stats)
            x = self.gemm(x)
            x = self.bn(x)
            x = self.scale * x
            x = nn.functional.softmax(x, dim=1)
            return x
        else:
            # Fused eval mode
            return self.fused_ops.fused_gemm_bn_scale_softmax_cuda(
                x,
                self.gemm.weight,
                self.gemm.bias,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.weight,
                self.bn.bias,
                self.scale,
                self.bn.eps
            )