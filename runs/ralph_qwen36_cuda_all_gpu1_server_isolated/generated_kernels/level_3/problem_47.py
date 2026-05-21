import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels for optimized operations
# We will fuse: Matmul(assignment), BatchNorm, Softmax, and the final normalization steps.
# However, to keep it robust and clear, we will implement specific high-impact kernels:
# 1. A fused kernel for Assignment Generation (Matmul + BatchNorm + Softmax)
# 2. A fused kernel for VLAD computation (Subtraction of mean vectors from cluster assignments)

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Helper for atomic add if needed, but here we use grid-stride loops or simple indexing

// Kernel 1: Compute Assignment Matrix (Matmul + BatchNorm + Softmax)
// Input: x (BN x D), clusters (D x K_total)
// Output: assignment (BN x K_total)
__global__ void compute_assignment_kernel(
    const float* __restrict__ x, 
    const float* __restrict__ clusters, 
    const float* __restrict__ bn_weight, 
    const float* __restrict__ bn_bias, 
    const float* __restrict__ bn_mean, 
    const float* __restrict__ bn_var_inv_sqrt,
    float* __restrict__ assignment, 
    int batch_size_times_n, 
    int feature_size, 
    int clusters_total) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size_times_n) return;

    // Each thread handles one sample in the flattened BN dimension
    // We need to compute dot products for all K clusters
    
    // Load x vector into shared memory? 
    // Given D=512, it fits in registers or local memory efficiently.
    // However, broadcasting clusters is better if we tile.
    // For simplicity and correctness with large D, we do direct computation.
    
    float sum = 0.0f;
    const float* x_ptr = x + idx * feature_size;
    
    // Compute dot product for all clusters
    // This part is memory bound if not optimized, but let's write clean code first.
    // To optimize, we can use a loop over D.
    for (int k = 0; k < clusters_total; ++k) {
        float dot = 0.0f;
        const float* c_ptr = clusters + k * feature_size;
        #pragma unroll
        for (int d = 0; d < feature_size; ++d) {
            dot += x_ptr[d] * c_ptr[d];
        }
        
        // BatchNorm: y = gamma * (x - mu) / sqrt(sigma^2 + eps) + beta
        float normalized = (dot - bn_mean[k]) * bn_var_inv_sqrt[k];
        if (bn_weight != nullptr) {
            normalized = normalized * bn_weight[k] + bn_bias[k];
        } else {
            normalized = normalized + bn_bias[k];
        }
        
        assignment[idx * clusters_total + k] = normalized;
    }
}

// Kernel 2: Softmax along dim 1 (K dimension)
__global__ void softmax_kernel(
    const float* __restrict__ input, 
    float* __restrict__ output, 
    int batch_size_times_n, 
    int clusters_total) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size_times_n) return;

    const float* row_ptr = input + idx * clusters_total;
    
    // Find max for numerical stability
    float max_val = -INFINITY;
    for (int k = 0; k < clusters_total; ++k) {
        if (row_ptr[k] > max_val) {
            max_val = row_ptr[k];
        }
    }
    
    // Compute exp and sum
    float sum_exp = 0.0f;
    for (int k = 0; k < clusters_total; ++k) {
        float val = expf(row_ptr[k] - max_val);
        output[idx * clusters_total + k] = val;
        sum_exp += val;
    }
    
    // Normalize
    float inv_sum = 1.0f / sum_exp;
    for (int k = 0; k < clusters_total; ++k) {
        output[idx * clusters_total + k] *= inv_sum;
    }
}

// Kernel 3: Compute VLAD residuals
// vlad[b, d, k] = sum_n (assignment[b, n, k] * x[b, n, d]) - a[b, d, k]
// where a[b, d, k] = (sum_n assignment[b, n, k]) * clusters2[d, k]
// Inputs: 
//   assignment: B x N x K (float)
//   x: B x N x D (float)
//   clusters2: 1 x D x K (float)
// Output: vlad: B x D x K (float)

__global__ void compute_vlad_residuals_kernel(
    const float* __restrict__ assignment, 
    const float* __restrict__ x, 
    const float* __restrict__ clusters2, 
    float* __restrict__ vlad, 
    int batch_size, 
    int num_features, 
    int cluster_size, 
    int feature_size) 
{
    // We launch one thread per output element: B * D * K
    int total_elements = batch_size * feature_size * cluster_size;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_elements) return;
    
    // Decode index to b, d, k
    int temp = idx;
    int k = temp % cluster_size;
    temp /= cluster_size;
    int d = temp % feature_size;
    int b = temp / feature_size;
    
    float sum_val = 0.0f;
    
    // Sum over N (num_features)
    // assignment is B x N x K, so index is b*N*K + n*K + k
    // x is B x N x D, so index is b*N*D + n*D + d
    
    const float* a_ptr = assignment + b * num_features * cluster_size;
    const float* x_ptr = x + b * num_features * feature_size;
    
    for (int n = 0; n < num_features; ++n) {
        sum_val += a_ptr[n * cluster_size + k] * x_ptr[n * feature_size + d];
    }
    
    // Subtract mean vector contribution
    // a_sum[b, 1, k] is not explicitly stored as a separate tensor in the original code 
    // but calculated via th.sum. Here we compute it on the fly or precompute?
    // The original code: a = a_sum * clusters2. 
    // a_sum[b, k] = sum_n assignment[b, n, k].
    // So we need to subtract (sum_n assignment[b, n, k]) * clusters2[d, k].
    
    float a_sum_k = 0.0f;
    for (int n = 0; n < num_features; ++n) {
        a_sum_k += a_ptr[n * cluster_size + k];
    }
    
    float mean_contribution = a_sum_k * clusters2[d * cluster_size + k];
    
    vlad[idx] = sum_val - mean_contribution;
}

// Kernel 4: L2 Normalization for VLAD (B x D x K) -> B x D x K
__global__ void l2_normalize_vlad_kernel(
    const float* __restrict__ input, 
    float* __restrict__ output, 
    int batch_size_times_dk, 
    int dk_dim) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size_times_dk) return;
    
    // Each thread handles one sample in the flattened B x DK dimension?
    // No, L2 norm is over the last dimension DK.
    // So we need to group by batch index.
    // Let's launch threads such that each block computes norm for one sample.
    // But standard grid-stride loop per element is easier if we handle reduction carefully.
    // Actually, let's use a simpler approach: 
    // Launch 1 thread per element, but we need atomic adds or shared memory for reduction.
    // Given DK can be large (32*512=16384), shared memory is good.
    
    // Alternative: Use a separate kernel that processes one sample at a time?
    // Let's stick to element-wise if we assume the caller handles the norm logic via a reduction kernel.
    // To keep it simple and robust, let's implement a standard L2 norm kernel that works on B x DK.
    
    // We will launch num_samples blocks, each block has dk_dim threads? No, too many threads.
    // Let's use a grid-stride loop where each thread computes its part of the sum of squares.
    
    int sample_idx = idx / dk_dim;
    int local_idx = idx % dk_dim;
    
    // This approach is tricky with simple indexing. 
    // Let's change strategy: Launch one block per sample? No, dynamic parallelism or too many blocks.
    // Let's use a standard reduction pattern inside the kernel if possible, or just do it in Python for norm?
    // The prompt asks for CUDA operators. Normalization is often fast enough in PyTorch, but let's optimize the heavy lifting.
    
    // Actually, let's just output the raw vlad and let PyTorch handle the final normalize, 
    // OR implement a simple element-wise kernel if we precompute norms.
    // Let's skip the L2 norm kernel for now to ensure compilation stability, as it requires complex reduction logic in inline CUDA.
    // The main bottleneck is Matmul and VLAD accumulation.
}

// Wrapper functions for PyTorch

torch::Tensor compute_assignment_cuda(
    torch::Tensor x, 
    torch::Tensor clusters, 
    torch::Tensor bn_weight, 
    torch::Tensor bn_bias, 
    torch::Tensor bn_mean, 
    torch::Tensor bn_var_inv_sqrt) 
{
    auto batch_size_times_n = x.size(0);
    auto feature_size = x.size(1);
    auto clusters_total = clusters.size(1);
    
    auto assignment = torch::zeros({batch_size_times_n, clusters_total}, x.options());
    
    const int block_size = 256;
    const int num_blocks = (batch_size_times_n + block_size - 1) / block_size;
    
    compute_assignment_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), 
        clusters.data_ptr<float>(), 
        bn_weight.data_ptr<float>(), 
        bn_bias.data_ptr<float>(), 
        bn_mean.data_ptr<float>(), 
        bn_var_inv_sqrt.data_ptr<float>(),
        assignment.data_ptr<float>(), 
        batch_size_times_n, 
        feature_size, 
        clusters_total
    );
    
    return assignment;
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    auto batch_size_times_n = input.size(0);
    auto clusters_total = input.size(1);
    
    auto output = torch::zeros_like(input);
    
    const int block_size = 256;
    const int num_blocks = (batch_size_times_n + block_size - 1) / block_size;
    
    softmax_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        batch_size_times_n, 
        clusters_total
    );
    
    return output;
}

torch::Tensor compute_vlad_residuals_cuda(
    torch::Tensor assignment, 
    torch::Tensor x, 
    torch::Tensor clusters2) 
{
    auto batch_size = assignment.size(0);
    auto num_features = assignment.size(1);
    auto cluster_size = assignment.size(2);
    auto feature_size = x.size(2);
    
    // Output shape: B x D x K
    auto vlad = torch::zeros({batch_size, feature_size, cluster_size}, x.options());
    
    int total_elements = batch_size * feature_size * cluster_size;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    compute_vlad_residuals_kernel<<<num_blocks, block_size>>>(
        assignment.data_ptr<float>(), 
        x.data_ptr<float>(), 
        clusters2.data_ptr<float>(), 
        vlad.data_ptr<float>(), 
        batch_size, 
        num_features, 
        cluster_size, 
        feature_size
    );
    
    return vlad;
}

"""

cuda_cpp_source = """
torch::Tensor compute_assignment_cuda(
    torch::Tensor x, 
    torch::Tensor clusters, 
    torch::Tensor bn_weight, 
    torch::Tensor bn_bias, 
    torch::Tensor bn_mean, 
    torch::Tensor bn_var_inv_sqrt);

torch::Tensor softmax_cuda(torch::Tensor input);

torch::Tensor compute_vlad_residuals_cuda(
    torch::Tensor assignment, 
    torch::Tensor x, 
    torch::Tensor clusters2);
"""

# Load the inline CUDA extension
optimized_ops = load_inline(
    name="optimized_vlad_ops",
    cpp_sources=cuda_cpp_source,
    cuda_sources=cuda_source,
    functions=[
        "compute_assignment_cuda",
        "softmax_cuda",
        "compute_vlad_residuals_cuda"
    ],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    def __init__(self, cluster_size, feature_size, ghost_clusters):
        super(ModelNew, self).__init__()

        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.ghost_clusters = ghost_clusters

        init_sc = (1 / math.sqrt(feature_size))
        clusters = cluster_size + ghost_clusters

        # The `clusters` weights are the `(w,b)` in the paper
        self.clusters = nn.Parameter(init_sc * th.randn(feature_size, clusters))
        
        # BatchNorm parameters
        self.bn_weight = nn.Parameter(th.ones(clusters))
        self.bn_bias = nn.Parameter(th.zeros(clusters))
        self.register_buffer('bn_mean', th.zeros(clusters))
        self.register_buffer('bn_var_inv_sqrt', th.ones(clusters))
        
        # The `clusters2` weights are the visual words `c_k` in the paper
        self.clusters2 = nn.Parameter(init_sc * th.randn(1, feature_size, cluster_size))
        
        self.out_dim = self.cluster_size * feature_size
        
        # Store references to custom ops
        self.compute_assignment_cuda = optimized_ops.compute_assignment_cuda
        self.softmax_cuda = optimized_ops.softmax_cuda
        self.compute_vlad_residuals_cuda = optimized_ops.compute_vlad_residuals_cuda

    def forward(self, x, mask=None):
        """Aggregates feature maps into a fixed size representation."""
        max_sample = x.size()[1]
        # Reshape to BN x D
        x_flat = x.view(-1, self.feature_size)  # B*N x D
        
        if x_flat.device != self.clusters.device:
            msg = f"x.device {x_flat.device} != cluster.device {self.clusters.device}"
            raise ValueError(msg)

        # Step 1: Compute Assignment (Matmul + BatchNorm) using custom CUDA
        # assignment shape: BN x (K+G)
        assignment_raw = self.compute_assignment_cuda(
            x_flat, 
            self.clusters, 
            self.bn_weight, 
            self.bn_bias, 
            self.bn_mean, 
            self.bn_var_inv_sqrt
        )

        # Step 2: Softmax using custom CUDA
        # assignment shape: BN x (K+G)
        assignment_soft = self.softmax_cuda(assignment_raw)

        # Remove ghost assignments
        assignment_soft = assignment_soft[:, :self.cluster_size]
        
        # Reshape to B x N x K
        assignment_soft = assignment_soft.view(-1, max_sample, self.cluster_size)
        
        # Compute a_sum: B x 1 x K
        a_sum = th.sum(assignment_soft, dim=1, keepdim=True)
        
        # Compute mean vector contribution: B x D x K (broadcasted)
        # clusters2 is 1 x D x K
        # a_sum is B x 1 x K
        # We need to multiply them. 
        # In the original code: a = a_sum * self.clusters2
        # This results in B x D x K if we broadcast correctly?
        # Original: assignment (BxNxK) -> sum dim 1 -> (Bx1xK). 
        # clusters2 is (1xDxK). 
        # Multiplication: (Bx1xK) * (1xDxK) -> B x D x K.
        
        a = a_sum * self.clusters2  # B x D x K

        # Transpose assignment to B x K x N for matmul with x (B x N x D)
        assignment_transposed = assignment_soft.transpose(1, 2)  # B x K x N
        
        # Reshape x back to B x N x D
        x_reshaped = x.view(-1, max_sample, self.feature_size)  # B x N x D
        
        # Compute VLAD: (B x K x N) x (B x N x D) -> B x K x D
        # Then transpose to B x D x K
        # vlad_raw = th.matmul(assignment_transposed, x_reshaped).transpose(1, 2)
        
        # Instead of standard matmul + transpose, we use the custom fused kernel
        # compute_vlad_residuals_cuda computes: sum_n (assign * x) - a
        # It expects assignment BxNxK, x BxNxD, clusters2 1xDxK
        # And outputs BxDxK
        
        vlad = self.compute_vlad_residuals_cuda(assignment_soft, x_reshaped, self.clusters2)
        
        # L2 intra norm (over D and K dimensions for each sample)
        # vlad is B x D x K. Flatten to B x DK
        vlad_flat = vlad.reshape(-1, self.cluster_size * self.feature_size)  # B x DK
        
        # Final L2 normalization
        vlad_norm = F.normalize(vlad_flat, p=2, dim=1)
        
        return vlad_norm

batch_size = 2048
num_features = 100
num_clusters = 32
feature_size = 512
ghost_clusters = 0

def get_inputs():
  return [torch.rand(batch_size, num_features, feature_size)]

def get_init_inputs():
  return [num_clusters, feature_size, ghost_clusters]