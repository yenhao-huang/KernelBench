import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source code for fused cross-entropy loss
cross_entropy_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void cross_entropy_loss_kernel(
    const float* __restrict__ predictions,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    int batch_size,
    int num_classes) {

    extern __shared__ float shared_mem[];
    float* s_max = shared_mem;          // size blockDim
    float* s_sum = shared_mem + blockDim.x; // size blockDim

    int row = blockIdx.x;
    if (row >= batch_size) return;

    const float* row_preds = predictions + row * num_classes;
    int tid = threadIdx.x;

    // Step 1: find max value in the row for numerical stability
    float local_max = -1e30f;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        float val = row_preds[i];
        if (val > local_max) local_max = val;
    }
    s_max[tid] = local_max;
    __syncthreads();

    // Block reduction for max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_max[tid + s] > s_max[tid]) s_max[tid] = s_max[tid + s];
        }
        __syncthreads();
    }
    float row_max = s_max[0];
    __syncthreads();

    // Step 2: compute sum of exp(x - max)
    float local_sum = 0.0f;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        float val = row_preds[i];
        local_sum += expf(val - row_max);
    }
    s_sum[tid] = local_sum;
    __syncthreads();

    // Block reduction for sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }
    float row_sum = s_sum[0];
    __syncthreads();

    // Step 3: get target logit and compute per-sample loss
    int target_idx = static_cast<int>(targets[row]);
    float target_logit = row_preds[target_idx];

    // Loss = log(sum) - (target_logit - max)
    float loss_val = logf(row_sum) - (target_logit - row_max);
    losses[row] = loss_val;
}

torch::Tensor cross_entropy_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.device().is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.device().is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");
    TORCH_CHECK(targets.dim() == 1, "targets must be 1D");
    TORCH_CHECK(predictions.size(0) == targets.size(0), "batch size mismatch");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kInt64, "targets must be int64");

    int batch_size = predictions.size(0);
    int num_classes = predictions.size(1);

    auto losses = torch::empty({batch_size}, predictions.options());

    const int block_size = 256;
    const int grid_size = batch_size;
    const int shared_mem_bytes = 2 * block_size * sizeof(float);

    cross_entropy_loss_kernel<<<grid_size, block_size, shared_mem_bytes>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        batch_size,
        num_classes
    );

    cudaDeviceSynchronize();
    return losses;
}
"""

cross_entropy_cpp_source = (
    "torch::Tensor cross_entropy_loss_cuda(torch::Tensor predictions, torch::Tensor targets);"
)

# Compile the inline CUDA code
cross_entropy_module = load_inline(
    name="cross_entropy_loss_cuda",
    cpp_sources=cross_entropy_cpp_source,
    cuda_sources=cross_entropy_cuda_source,
    functions=["cross_entropy_loss_cuda"],
    verbose=False,
    extra_cflags=["--use_fast_math"],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.cross_entropy = cross_entropy_module

    def forward(self, predictions, targets):
        per_sample_losses = self.cross_entropy.cross_entropy_loss_cuda(predictions, targets)
        return per_sample_losses.mean()