import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cpp_sources = """
torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets);
"""

cuda_sources = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hinge_sum_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ partial,
    long long total,
    int ncols
) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    long long idx = (long long)blockIdx.x * blockDim.x + tid;
    long long stride = (long long)blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (long long k = idx; k < total; k += stride) {
        float v = 1.0f - predictions[k] * targets[k % ncols];
        sum += v > 0.0f ? v : 0.0f;
    }

    smem[tid] = sum;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) smem[tid] += smem[tid + offset];
        __syncthreads();
    }

    if (tid == 0) partial[blockIdx.x] = smem[0];
}

__global__ void hinge_finalize_kernel(
    const float* __restrict__ partial,
    float* __restrict__ out,
    int blocks,
    float inv_total
) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;

    float sum = 0.0f;
    for (int i = tid; i < blocks; i += blockDim.x) {
        sum += partial[i];
    }

    smem[tid] = sum;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) smem[tid] += smem[tid + offset];
        __syncthreads();
    }

    if (tid == 0) out[0] = smem[0] * inv_total;
}

torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    const long long total = predictions.numel();
    const int ncols = targets.numel();

    const int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 4096) blocks = 4096;
    if (blocks < 1) blocks = 1;

    auto partial = torch::empty({blocks}, predictions.options());
    auto out = torch::empty({}, predictions.options());

    hinge_sum_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        partial.data_ptr<float>(),
        total,
        ncols
    );

    hinge_finalize_kernel<<<1, 1024, 1024 * sizeof(float)>>>(
        partial.data_ptr<float>(),
        out.data_ptr<float>(),
        blocks,
        1.0f / (float)total
    );

    return out;
}
"""

hinge_loss_ext = load_inline(
    name="hinge_loss_ext_fast",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["hinge_loss_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.hinge_loss_ext = hinge_loss_ext

    def forward(self, predictions, targets):
        return self.hinge_loss_ext.hinge_loss_cuda(predictions, targets)