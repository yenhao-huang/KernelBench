import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused NetVLAD aggregation
# This kernel computes: vlad = (assignment^T @ x) - (a_sum * clusters2)
# where assignment is B x N x K, x is B x N x D, a_sum is B x 1 x K, and clusters2 is 1 x D x K.
vlad_aggregation_cpp_source = """
torch::Tensor vlad_aggregation_cuda(
    torch::Tensor x, 
    torch::Tensor assignment, 
    torch::Tensor clusters2, 
    torch::Tensor a_sum);
"""

vlad_aggregation_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void vlad_aggregation_kernel(
    const float* __restrict__ x,      // B * N * D
    const float* __restrict__ assignment, // B * N * K
    const float* __restrict__ clusters2, // D * K
    const float* __restrict__ a_sum,    // B * K
    float* __restrict__ vlad,          // B * D * K
    int B, int N, int D, int K) {
    
    int k = blockIdx.x; 
    int d = blockIdx.y; 
    int b = blockIdx.z; 
    
    if (k < K && d < D && b < B) {
        float sum = 0.0f;
        // The loop over N is the bottleneck, but it's necessary for the aggregation.
        // This kernel is memory-bound.
        for (int n = 0; n < N; ++n) {
            sum += assignment[b * N * K + n * K + k] * x[b * N * D + n * D + d];
        }
        float a_val = a_sum[b * K + k] * clusters2[d * K + k];
        vlad[b * D * K + d * K + k] = sum - a_val;
    }
}

torch::Tensor vlad_aggregation_cuda(
    torch::Tensor x, 
    torch::Tensor assignment, 
    torch::Tensor clusters2, 
    torch::Tensor a_sum) {
    
    int B = x.size(0);
    int N = x.size(1);
    int D = x.size(2);
    int K = assignment.size(2);

    auto vlad = torch::empty({B, D, K}, x.options());

    dim3 block(1, 1, 1);
    dim3 grid(K, D, B);

    vlad_aggregation_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        assignment.data_ptr<float>(),
        clusters2.data_ptr<float>(),
        a_sum.data_ptr<float>(),
        vlad.data_ptr<float>(),
        B, N, D, K
    );

    return vlad;
}
"""

# Compile the inline CUDA code
vlad_ops = load_inline(
    name="vlad_ops",
    cpp_sources=vlad_aggregation_cpp_source,
    cuda_sources=vlad_aggregation_cuda_source,
    functions=["vlad_aggregation_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, cluster_size, feature_size, ghost_clusters):
        super(ModelNew, self).__init__()

        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.ghost_clusters = ghost_clusters

        init_sc = (1 / math.sqrt(feature_size))
        clusters_count = cluster_size + ghost_clusters

        self.clusters = nn.Parameter(init_sc * torch.randn(feature_size, clusters_count))
        self.batch_norm = nn.BatchNorm1d(clusters_count)
        self.clusters2 = nn.Parameter(init_sc * torch.randn(1, feature_size, cluster_size))
        self.out_dim = self.cluster_size * feature_size
        self.vlad_ops = vlad_ops

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        max_sample = x.size(1)
        
        # 1. Assignment calculation
        # x: B x N x D -> BN x D
        x_flat = x.view(-1, self.feature_size)
        
        # assignment: BN x (K+G)
        assignment = torch.matmul(x_flat, self.clusters)
        assignment = self.batch_norm(assignment)
        assignment = F.softmax(assignment, dim=1)
        
        # 2. Remove ghost clusters and prepare for aggregation
        # assignment: BN x K
        assignment = assignment[:, :self.cluster_size]
        
        # Reshape assignment to B x N x K
        assignment_bnk = assignment.view(batch_size, max_sample, self.cluster_size)
        
        # a_sum: B x 1 x K
        a_sum = torch.sum(assignment_bnk, dim=1, keepdim=True)
        
        # Prepare inputs for custom kernel
        # a_sum needs to be B x K for the kernel
        a_sum_flat = a_sum.view(batch_size, self.cluster_size)
        
        # clusters2 is 1 x D x K, kernel expects D x K
        clusters2_flat = self.clusters2.view(self.feature_size, self.cluster_size)
        
        # 3. Fused Aggregation: vlad = (assignment^T @ x) - (a_sum * clusters2)
        # x is B x N x D
        vlad = self.vlad_ops.vlad_aggregation_cuda(
            x, 
            assignment_bnk, 
            clusters2_flat, 
            a_sum_flat
        )
        
        # 4. L2 intra norm
        # vlad is B x D x K. Normalize over D (dim=1)
        vlad = F.normalize(vlad, p=2, dim=1)

        # 5. Flattening + L2 norm
        # vlad: B x DK
        vlad = vlad.reshape(batch_size, -1)
        vlad = F.normalize(vlad, p=2, dim=1)
        
        return vlad