import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for fused MSE loss: (x - y)^2 mean
mse_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mse_kernel(const float* __restrict__ pred, const float* __restrict__ target, float* __restrict__ global_sum, int N) {
    __shared__ float sdata[256];
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    float my_sum = 0.0f;
    // Grid-stride loop to handle arbitrary N
    for (int i = idx; i < N; i += gridDim.x * blockDim.x) {
        float diff = pred[i] - target[i];
        my_sum += diff * diff;
    }
    sdata[tid] = my_sum;
    __syncthreads();
    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(global_sum, sdata[0]);
}

torch::Tensor mse_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    // Ensure inputs are contiguous float tensors on the same device
    auto pred = predictions.contiguous();
    auto targ = targets.contiguous();
    TORCH_CHECK(pred.sizes() == targ.sizes(), "Input tensors must have the same shape");
    int64_t N = pred.numel();
    auto options = torch::TensorOptions().device(pred.device()).dtype(torch::kFloat32);
    auto global_sum = torch::zeros({1}, options);

    const int block_size = 256;
    int num_blocks = (N + block_size - 1) / block_size;
    // Cap blocks to avoid launching too many (still efficient with loop)
    num_blocks = std::min(num_blocks, 65535);
    if (num_blocks < 1) num_blocks = 1;

    mse_kernel<<<num_blocks, block_size>>>(
        pred.data_ptr<float>(), targ.data_ptr<float>(), global_sum.data_ptr<float>(), N);

    // Compute mean
    auto mean = global_sum / static_cast<float>(N);
    return mean;
}
"""

mse_loss_cpp_source = "torch::Tensor mse_loss_cuda(torch::Tensor predictions, torch::Tensor targets);"

# Compile the custom operator
mse_loss = load_inline(
    name="mse_loss",
    cpp_sources=mse_loss_cpp_source,
    cuda_sources=mse_loss_source,
    functions=["mse_loss_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mse_loss = mse_loss

    def forward(self, predictions, targets):
        return self.mse_loss.mse_loss_cuda(predictions, targets)