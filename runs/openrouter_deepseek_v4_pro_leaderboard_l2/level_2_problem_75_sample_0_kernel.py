import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused GEMM + GroupNorm + Min + Bias
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Fused kernel: GEMM (Linear) + GroupNorm + Min reduction + Bias addition
// Input: x [batch_size, in_features]
// Weight: weight [out_features, in_features]
// Bias (gemm): gemm_bias [out_features] (optional, but Linear has bias)
// GroupNorm: gamma [out_features], beta [out_features]
// Bias (final): bias [1, out_features, 1, 1] -> treated as [out_features]
// Output: result [batch_size, 1] (min over dim=1, keepdim=True, then + bias)

// We'll compute the entire pipeline in one kernel per batch element.
// For each batch element, we compute the GEMM output (out_features), then group norm,
// then find the minimum, then add the final bias.

__global__ void fused_gemm_groupnorm_min_bias_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ gemm_bias,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ final_bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    int num_groups
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    int group_size = out_features / num_groups;
    int tid = threadIdx.x;
    int threads_per_block = blockDim.x;

    // Shared memory for GEMM output of this batch element
    extern __shared__ float shared_mem[];
    float* gemm_out = shared_mem; // size out_features
    float* group_mean = gemm_out + out_features; // size num_groups
    float* group_var = group_mean + num_groups; // size num_groups

    // Initialize gemm_out to gemm_bias (if present) or zero
    // We'll do this cooperatively
    for (int i = tid; i < out_features; i += threads_per_block) {
        gemm_out[i] = (gemm_bias != nullptr) ? gemm_bias[i] : 0.0f;
    }
    __syncthreads();

    // Compute GEMM: gemm_out += x[batch_idx, :] * weight[:, :]
    // Each thread computes partial dot products for multiple output features
    // We'll use a tiled approach with shared memory for x to reduce global memory reads.
    // But for simplicity, we can just have each thread handle a chunk of in_features for a subset of out_features.
    // Since out_features=8192, in_features=8192, we can assign each thread to compute one output element.
    // But we have limited threads (e.g., 256). So each thread computes multiple output elements.
    // We'll use a loop over output features.

    // For each output feature j that this thread is responsible for:
    for (int j = tid; j < out_features; j += threads_per_block) {
        float sum = gemm_out[j];
        const float* x_ptr = x + batch_idx * in_features;
        const float* w_ptr = weight + j * in_features;
        for (int k = 0; k < in_features; ++k) {
            sum += x_ptr[k] * w_ptr[k];
        }
        gemm_out[j] = sum;
    }
    __syncthreads();

    // Now compute group norm statistics per group
    // Each thread handles one group (or multiple groups)
    for (int g = tid; g < num_groups; g += threads_per_block) {
        int start = g * group_size;
        int end = start + group_size;
        float mean = 0.0f;
        for (int i = start; i < end; ++i) {
            mean += gemm_out[i];
        }
        mean /= group_size;
        group_mean[g] = mean;

        float var = 0.0f;
        for (int i = start; i < end; ++i) {
            float diff = gemm_out[i] - mean;
            var += diff * diff;
        }
        var = var / group_size;
        group_var[g] = var;
    }
    __syncthreads();

    // Apply normalization and compute min per batch element
    // We'll compute min using parallel reduction within the block.
    // First, each thread computes normalized values for its assigned output features,
    // then we find the minimum among them.

    // We'll use shared memory for reduction of min.
    // We need to find the minimum over all out_features normalized values.
    // We can do a block-level reduction.

    // Each thread computes normalized values for its assigned j, and keeps track of local min.
    float local_min = INFINITY;
    for (int j = tid; j < out_features; j += threads_per_block) {
        int g = j / group_size;
        float mean = group_mean[g];
        float var = group_var[g];
        float inv_std = rsqrtf(var + 1e-5f);
        float val = (gemm_out[j] - mean) * inv_std;
        // Apply gamma and beta
        val = val * gamma[j] + beta[j];
        if (val < local_min) local_min = val;
    }

    // Now reduce local_min across all threads in the block to find global min for this batch element.
    // We'll use shared memory for reduction.
    // We need a shared array for min reduction. We can reuse part of shared_mem.
    // We'll use the first threads_per_block floats of shared_mem for reduction.
    float* min_shared = shared_mem; // reuse gemm_out space, but careful: we still need gemm_out? No, we are done with it.
    // Actually, we need to ensure no conflict. We'll use a separate part.
    // Let's use the space after group_var.
    float* min_reduce = group_var + num_groups; // size at least threads_per_block
    // Ensure we have enough shared memory allocated.
    // We'll allocate enough: out_features + 2*num_groups + threads_per_block.
    // But we already declared extern __shared__ float shared_mem[]; we'll just index carefully.

    min_reduce[tid] = local_min;
    __syncthreads();

    // Parallel reduction for min
    for (int stride = threads_per_block / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other = min_reduce[tid + stride];
            if (other < min_reduce[tid]) min_reduce[tid] = other;
        }
        __syncthreads();
    }

    // Thread 0 writes the final min + bias to output
    if (tid == 0) {
        float min_val = min_reduce[0];
        // Add final bias: bias is shape [1, out_features, 1, 1], but we need to add it to the min?
        // Wait, the original code: x = torch.min(x, dim=1, keepdim=True)[0] -> shape [batch_size, 1]
        // Then x = x + self.bias where bias shape is (1, out_features, 1, 1).
        // That addition broadcasts: [batch_size, 1] + [1, out_features, 1, 1] -> [batch_size, out_features, 1, 1]?
        // Actually, the original code has x shape after min: [batch_size, 1] (since keepdim=True, dim=1).
        // Then x + self.bias: bias shape (1, out_features, 1, 1). Broadcasting rules:
        // [batch_size, 1] + [1, out_features, 1, 1] -> [batch_size, out_features, 1, 1].
        // But the output of the model is that? The original forward returns x after bias addition.
        // So the output shape should be [batch_size, out_features, 1, 1].
        // However, the min operation reduces dim=1, so the result has size 1 in that dimension.
        // Then adding bias of shape (1, out_features, 1, 1) yields [batch_size, out_features, 1, 1].
        // That seems odd: min over features then add a per-feature bias? That would broadcast the min value to all features and add bias.
        // Let's check: x after group_norm: [batch_size, out_features]
        // min(x, dim=1, keepdim=True)[0] -> [batch_size, 1]
        // bias: [1, out_features, 1, 1]
        // x + bias: [batch_size, 1] + [1, out_features, 1, 1] = [batch_size, out_features, 1, 1]
        // So each batch element gets a vector of length out_features, where each element is min_val + bias[0, i, 0, 0].
        // That is the output.
        // So we need to output a tensor of shape [batch_size, out_features, 1, 1].
        // But our kernel currently only computes one min per batch. We need to write the result for each feature.
        // So we should write min_val + final_bias[j] for each j.
        // We can have thread 0 write all out_features values, or have each thread write its assigned j.
        // Since output is [batch_size, out_features, 1, 1], we can have each thread write its part.
        // But we already have the min_val. We'll let each thread write its assigned output features.
        // We'll store min_val in shared memory and then each thread writes.
    }

    // After reduction, thread 0 has the min. Broadcast min to all threads via shared memory.
    __syncthreads();
    float min_val = min_reduce[0];
    // Now each thread writes the output for its assigned features.
    // Output indexing: output[batch_idx * out_features + j] = min_val + final_bias[j]
    // But output shape is [batch_size, out_features, 1, 1], so we can treat it as 2D [batch_size, out_features].
    for (int j = tid; j < out_features; j += threads_per_block) {
        output[batch_idx * out_features + j] = min_val + final_bias[j];
    }
}

torch::Tensor fused_gemm_groupnorm_min_bias_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor gemm_bias,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor final_bias,
    int num_groups
) {
    int batch_size = x.size(0);
    int in_features = x.size(1);
    int out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features, 1, 1}, x.options());

    const int threads_per_block = 256;
    const int blocks = batch_size;

    int group_size = out_features / num_groups;
    // Shared memory: out_features floats for gemm_out, num_groups for mean, num_groups for var, threads_per_block for reduction
    int shared_mem_size = (out_features + 2 * num_groups + threads_per_block) * sizeof(float);

    fused_gemm_groupnorm_min_bias_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        gemm_bias.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        final_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        num_groups
    );

    return output;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_gemm_groupnorm_min_bias_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor gemm_bias,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor final_bias,
    int num_groups
);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_gemm_groupnorm_min_bias",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_gemm_groupnorm_min_bias_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_op = fused_op
        self.num_groups = num_groups

    def forward(self, x):
        # Use the fused CUDA kernel that performs GEMM, GroupNorm, Min, and Bias addition
        return self.fused_op.fused_gemm_groupnorm_min_bias_cuda(
            x,
            self.gemm.weight,
            self.gemm.bias,
            self.group_norm.weight,
            self.group_norm.bias,
            self.bias,
            self.num_groups
        )