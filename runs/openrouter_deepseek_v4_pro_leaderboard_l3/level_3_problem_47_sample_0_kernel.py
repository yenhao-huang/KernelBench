import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th
from torch.utils.cpp_extension import load_inline

# CUDA source for fused VLAD aggregation kernel
vlad_aggregation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void vlad_aggregation_kernel(
    const float* __restrict__ assignment,
    const float* __restrict__ x,
    const float* __restrict__ a_sum,
    const float* __restrict__ clusters2,
    float* __restrict__ output,
    int B, int N, int D, int K)
{
    extern __shared__ float s_v[];
    int b = blockIdx.x / K;
    int k = blockIdx.x % K;
    int d = threadIdx.x;

    float v = 0.0f;
    if (d < D) {
        for (int n = 0; n < N; ++n) {
            float assign_val = assignment[b * (N * K) + n * K + k];
            float x_val = x[b * (N * D) + n * D + d];
            v += assign_val * x_val;
        }
        float a = a_sum[b * K + k] * clusters2[d * K + k];
        v -= a;
        s_v[d] = v * v;
    } else {
        s_v[d] = 0.0f;
    }
    __syncthreads();

    // Parallel reduction to compute sum of squares
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (d < stride) {
            s_v[d] += s_v[d + stride];
        }
        __syncthreads();
    }

    float norm = sqrtf(s_v[0] + 1e-12f);
    float inv_norm = 1.0f / norm;

    if (d < D) {
        output[b * (D * K) + d * K + k] = v * inv_norm;
    }
}

torch::Tensor vlad_aggregation_cuda(
    torch::Tensor assignment,
    torch::Tensor x,
    torch::Tensor a_sum,
    torch::Tensor clusters2)
{
    int B = assignment.size(0);
    int N = assignment.size(1);
    int K = assignment.size(2);
    int D = x.size(2);

    auto output = torch::empty({B, D, K}, assignment.options());

    // Choose block size as next power of 2 >= D, capped at 1024
    int block_size = 1;
    while (block_size < D) block_size <<= 1;
    if (block_size > 1024) block_size = 1024;

    const int grid_size = B * K;
    size_t shared_mem_size = block_size * sizeof(float);

    vlad_aggregation_kernel<<<grid_size, block_size, shared_mem_size>>>(
        assignment.data_ptr<float>(),
        x.data_ptr<float>(),
        a_sum.data_ptr<float>(),
        clusters2.data_ptr<float>(),
        output.data_ptr<float>(),
        B, N, D, K);

    return output;
}
"""

vlad_aggregation_cpp_source = "torch::Tensor vlad_aggregation_cuda(torch::Tensor assignment, torch::Tensor x, torch::Tensor a_sum, torch::Tensor clusters2);"

# Compile the inline CUDA code
vlad_aggregation = load_inline(
    name="vlad_aggregation",
    cpp_sources=vlad_aggregation_cpp_source,
    cuda_sources=vlad_aggregation_source,
    functions=["vlad_aggregation_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, cluster_size, feature_size, ghost_clusters):
        super(ModelNew, self).__init__()

        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.ghost_clusters = ghost_clusters

        init_sc = (1 / math.sqrt(feature_size))
        clusters = cluster_size + ghost_clusters

        self.clusters = nn.Parameter(init_sc * th.randn(feature_size, clusters))
        self.batch_norm = nn.BatchNorm1d(clusters)
        self.clusters2 = nn.Parameter(init_sc * th.randn(1, feature_size, cluster_size))
        self.out_dim = self.cluster_size * feature_size

        # Custom CUDA operator for fused VLAD aggregation
        self.vlad_aggregation = vlad_aggregation

    def forward(self, x, mask=None):
        """Aggregates feature maps into a fixed size representation.

        Args:
            x (th.Tensor): B x N x D

        Returns:
            (th.Tensor): B x DK
        """
        max_sample = x.size()[1]
        x_flat = x.reshape(-1, self.feature_size)  # BN x D

        if x.device != self.clusters.device:
            msg = f"x.device {x.device} != cluster.device {self.clusters.device}"
            raise ValueError(msg)

        # First matmul + batch norm + softmax (kept as original)
        assignment = th.matmul(x_flat, self.clusters)  # BN x (K+G)
        assignment = self.batch_norm(assignment)
        assignment = F.softmax(assignment, dim=1)  # BN x (K+G)

        # Remove ghost clusters
        assignment = assignment[:, :self.cluster_size]  # BN x K
        assignment = assignment.view(-1, max_sample, self.cluster_size)  # B x N x K

        # Compute a_sum = sum over N
        a_sum = th.sum(assignment, dim=1, keepdim=True)  # B x 1 x K

        # Fused kernel: matmul(assignment^T, x) - a_sum * clusters2 + L2 normalize
        vlad = self.vlad_aggregation.vlad_aggregation_cuda(
            assignment.contiguous(),
            x.contiguous(),
            a_sum.contiguous(),
            self.clusters2.contiguous()
        )  # B x D x K

        # Flattening + final L2 norm
        vlad = vlad.reshape(-1, self.cluster_size * self.feature_size)  # B x DK
        vlad = F.normalize(vlad)
        return vlad