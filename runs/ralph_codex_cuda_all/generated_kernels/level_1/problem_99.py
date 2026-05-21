import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

triplet_loss_cpp = """
torch::Tensor triplet_margin_loss_cuda(torch::Tensor anchor, torch::Tensor positive, torch::Tensor negative, double margin);
"""

triplet_loss_cuda = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void triplet_margin_loss_kernel(
    const float* __restrict__ anchor,
    const float* __restrict__ positive,
    const float* __restrict__ negative,
    float* __restrict__ out,
    int batch,
    int dim,
    float margin
) {
    extern __shared__ float smem[];
    float* spos = smem;
    float* sneg = smem + blockDim.x;

    int row = blockIdx.x;
    int tid = threadIdx.x;

    float pos_sum = 0.0f;
    float neg_sum = 0.0f;
    int base = row * dim;

    for (int i = tid; i < dim; i += blockDim.x) {
        float ap = anchor[base + i] - positive[base + i] + 1.0e-6f;
        float an = anchor[base + i] - negative[base + i] + 1.0e-6f;
        pos_sum += ap * ap;
        neg_sum += an * an;
    }

    spos[tid] = pos_sum;
    sneg[tid] = neg_sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            spos[tid] += spos[tid + stride];
            sneg[tid] += sneg[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float loss = sqrtf(spos[0]) - sqrtf(sneg[0]) + margin;
        if (loss > 0.0f) {
            atomicAdd(out, loss / (float)batch);
        }
    }
}

torch::Tensor triplet_margin_loss_cuda(torch::Tensor anchor, torch::Tensor positive, torch::Tensor negative, double margin) {
    int batch = anchor.size(0);
    int dim = anchor.numel() / batch;

    auto out = torch::zeros({}, anchor.options());

    const int threads = 256;
    const dim3 blocks(batch);
    const size_t shmem = 2 * threads * sizeof(float);

    triplet_margin_loss_kernel<<<blocks, threads, shmem>>>(
        anchor.data_ptr<float>(),
        positive.data_ptr<float>(),
        negative.data_ptr<float>(),
        out.data_ptr<float>(),
        batch,
        dim,
        (float)margin
    );

    return out;
}
"""

_triplet_loss_mod = load_inline(
    name="triplet_margin_loss_custom_cuda",
    cpp_sources=triplet_loss_cpp,
    cuda_sources=triplet_loss_cuda,
    functions=["triplet_margin_loss_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = float(margin)
        self.triplet_loss = _triplet_loss_mod

    def forward(self, anchor, positive, negative):
        return self.triplet_loss.triplet_margin_loss_cuda(anchor, positive, negative, self.margin)