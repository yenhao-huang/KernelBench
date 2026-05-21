import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for the fused kernel
fused_kernel_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_matmul_avgpool_gelu_scale_max_kernel(
    const float* __restrict__ x,
    const float* __restrict__ group_weight_T,
    const float* __restrict__ group_bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int num_groups,
    float inv_pool,
    float scale_factor
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    extern __shared__ float shared[];
    float* x_shared = shared;
    float* max_shared = &shared[in_features];

    int tid = threadIdx.x;
    int block_dim = blockDim.x;

    // Load x for this batch into shared memory
    const float* x_batch = x + batch_idx * in_features;
    for (int i = tid; i < in_features; i += block_dim) {
        x_shared[i] = x_batch[i];
    }
    __syncthreads();

    // Each thread handles a contiguous chunk of groups
    int groups_per_thread = (num_groups + block_dim - 1) / block_dim;
    int start_g = tid * groups_per_thread;
    int end_g = min(start_g + groups_per_thread, num_groups);

    float local_max = -1e30f;

    for (int g = start_g; g < end_g; ++g) {
        float sum = group_bias[g];
        for (int i = 0; i < in_features; ++i) {
            sum += x_shared[i] * group_weight_T[i * num_groups + g];
        }
        float val = sum * inv_pool;
        // GELU activation
        float gelu = 0.5f * val * (1.0f + tanhf(0.7978845608028654f * (val + 0.044715f * val * val * val)));
        float scaled = gelu * scale_factor;
        local_max = fmaxf(local_max, scaled);
    }

    // Write local max to shared memory
    max_shared[tid] = local_max;
    __syncthreads();

    // Reduction to find global max
    for (int stride = block_dim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            max_shared[tid] = fmaxf(max_shared[tid], max_shared[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[batch_idx] = max_shared[0];
    }
}

torch::Tensor fused_matmul_avgpool_gelu_scale_max_cuda(
    torch::Tensor x,
    torch::Tensor group_weight_T,
    torch::Tensor group_bias,
    float inv_pool,
    float scale_factor
) {
    int batch_size = x.size(0);
    int in_features = x.size(1);
    int num_groups = group_bias.size(0);

    auto output = torch::empty({batch_size}, x.options());

    const int block_size = 256;
    const int grid_size = batch_size;
    size_t shared_mem_size = (in_features + block_size) * sizeof(float);

    fused_matmul_avgpool_gelu_scale_max_kernel<<<grid_size, block_size, shared_mem_size>>>(
        x.data_ptr<float>(),
        group_weight_T.data_ptr<float>(),
        group_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        num_groups,
        inv_pool,
        scale_factor
    );

    return output;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_matmul_avgpool_gelu_scale_max_cuda(
    torch::Tensor x,
    torch::Tensor group_weight_T,
    torch::Tensor group_bias,
    float inv_pool,
    float scale_factor
);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_matmul_avgpool_gelu_scale_max",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_cuda_source,
    functions=["fused_matmul_avgpool_gelu_scale_max_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.scale_factor = scale_factor
        self.inv_pool = 1.0 / pool_kernel_size

        # Original linear layer to obtain weight and bias
        linear = nn.Linear(in_features, out_features)
        weight = linear.weight.data  # shape: (out_features, in_features)
        bias = linear.bias.data      # shape: (out_features)

        num_groups = out_features // pool_kernel_size

        # Precompute grouped weights and biases
        # group_weight: (num_groups, in_features) = sum over pool_kernel_size
        group_weight = weight.view(num_groups, pool_kernel_size, in_features).sum(dim=1)
        # Transpose to (in_features, num_groups) for coalesced access
        group_weight_T = group_weight.transpose(0, 1).contiguous()
        group_bias = bias.view(num_groups, pool_kernel_size).sum(dim=1)

        # Register as buffers (non-trainable)
        self.register_buffer('group_weight_T', group_weight_T)
        self.register_buffer('group_bias', group_bias)

        # Keep the custom op
        self.fused_op = fused_op

    def forward(self, x):
        return self.fused_op.fused_matmul_avgpool_gelu_scale_max_cuda(
            x,
            self.group_weight_T,
            self.group_bias,
            self.inv_pool,
            self.scale_factor
        )