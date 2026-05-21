import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA implementation for Triplet Margin Loss
# This kernel fuses the distance calculation, margin application, and max/sum operations
triplet_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Helper to compute squared Euclidean distance between two vectors
__device__ inline float compute_sq_dist(const float* a, const float* b, int dim) {
    float dist = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

__global__ void triplet_loss_kernel(
    const float* anchor, 
    const float* positive, 
    const float* negative, 
    float* out, 
    int batch_size, 
    int dim, 
    float margin) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        const float* a_ptr = anchor + idx * dim;
        const float* p_ptr = positive + idx * dim;
        const float* n_ptr = negative + idx * dim;

        // Compute squared distances
        float dist_pos = compute_sq_dist(a_ptr, p_ptr, dim);
        float dist_neg = compute_sq_dist(a_ptr, n_ptr, dim);

        // Triplet loss: max(0, dist_pos - dist_neg + margin)
        float loss_val = dist_pos - dist_neg + margin;
        if (loss_val < 0.0f) {
            loss_val = 0.0f;
        }

        out[idx] = loss_val;
    }
}

torch::Tensor triplet_loss_cuda(torch::Tensor anchor, torch::Tensor positive, torch::Tensor negative, float margin) {
    TORCH_CHECK(anchor.is_cuda(), "anchor must be a CUDA tensor");
    TORCH_CHECK(positive.is_cuda(), "positive must be a CUDA tensor");
    TORCH_CHECK(negative.is_cuda(), "negative must be a CUDA tensor");
    
    TORCH_CHECK(anchor.sizes() == positive.sizes(), "anchor and positive must have same shape");
    TORCH_CHECK(anchor.sizes() == negative.sizes(), "anchor and negative must have same shape");

    auto batch_size = anchor.size(0);
    auto dim = anchor.numel() / batch_size;

    auto out = torch::zeros({batch_size}, anchor.options());

    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;

    triplet_loss_kernel<<<num_blocks, block_size>>>(
        anchor.data_ptr<float>(), 
        positive.data_ptr<float>(), 
        negative.data_ptr<float>(), 
        out.data_ptr<float>(), 
        batch_size, 
        dim, 
        margin
    );

    return out;
}
"""

triplet_loss_cpp_source = (
    "torch::Tensor triplet_loss_cuda(torch::Tensor anchor, torch::Tensor positive, torch::Tensor negative, float margin);"
)

# Compile the inline CUDA code
triplet_loss_module = load_inline(
    name="triplet_loss_custom",
    cpp_sources=triplet_loss_cpp_source,
    cuda_sources=triplet_loss_source,
    functions=["triplet_loss_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    A model that computes Triplet Margin Loss for metric learning tasks using custom CUDA.

    Parameters:
        margin (float): The margin between the positive and negative samples.
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        return triplet_loss_module.triplet_loss_cuda(anchor, positive, negative, self.margin)