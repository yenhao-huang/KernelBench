import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA source code with all custom kernels
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void softmax_slice_kernel(const float* input, float* output, int BN, int K_G, int K) {
    int row = blockIdx.x;
    if (row >= BN) return;
    int lane = threadIdx.x;
    
    // Compute max
    float max_val = -INFINITY;
    for (int i = lane; i < K_G; i += 32) {
        float val = input[row * K_G + i];
        if (val > max_val) max_val = val;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, max_val, offset);
        if (other > max_val) max_val = other;
    }
    max_val = __shfl_sync(0xffffffff, max_val, 0);
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = lane; i < K_G; i += 32) {
        float val = expf(input[row * K_G + i] - max_val);
        sum += val;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    sum = __shfl_sync(0xffffffff, sum, 0);
    
    // Write sliced output (first K elements)
    for (int i = lane; i < K; i += 32) {
        float val = expf(input[row * K_G + i] - max_val) / sum;
        output[row * K + i] = val;
    }
}

torch::Tensor softmax_slice_cuda(torch::Tensor input, int K) {
    int BN = input.size(0);
    int K_G = input.size(1);
    auto output = torch::empty({BN, K}, input.options());
    
    const int threads = 32;
    const int blocks = BN;
    
    softmax_slice_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), BN, K_G, K
    );
    return output;
}

__global__ void compute_a_kernel(const float* softmax_sliced, const float* clusters2, float* a,
                                 int B, int N, int D, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * D * K;
    if (idx >= total) return;
    
    int b = idx / (D * K);
    int rem = idx % (D * K);
    int d = rem / K;
    int k = rem % K;
    
    float sum = 0.0f;
    for (int n = 0; n < N; ++n) {
        sum += softmax_sliced[b * N * K + n * K + k];
    }
    a[idx] = sum * clusters2[d * K + k];
}

torch::Tensor compute_a_cuda(torch::Tensor softmax_sliced, torch::Tensor clusters2, int B, int N, int D, int K) {
    auto a = torch::empty({B, D, K}, softmax_sliced.options());
    
    int total = B * D * K;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    
    compute_a_kernel<<<blocks, threads>>>(
        softmax_sliced.data_ptr<float>(), clusters2.data_ptr<float>(), a.data_ptr<float>(),
        B, N, D, K
    );
    return a;
}

__global__ void vlad_subtract_intranorm_kernel(const float* vlad, const float* a, float* out,
                                               int B, int D, int K) {
    int bk = blockIdx.x;
    int b = bk / K;
    int k = bk % K;
    if (b >= B) return;
    
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int d = tid; d < D; d += stride) {
        int idx = b * D * K + d * K + k;
        float val = vlad[idx] - a[idx];
        sum_sq += val * val;
    }
    
    __shared__ float shared_sum[256];
    shared_sum[tid] = sum_sq;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    float norm = sqrtf(shared_sum[0] + 1e-12f);
    
    // Normalize and write output
    for (int d = tid; d < D; d += stride) {
        int idx = b * D * K + d * K + k;
        float val = vlad[idx] - a[idx];
        out[idx] = val / norm;
    }
}

torch::Tensor vlad_subtract_intranorm_cuda(torch::Tensor vlad, torch::Tensor a, int B, int D, int K) {
    auto out = torch::empty_like(vlad);
    
    int blocks = B * K;
    const int threads = 256;
    
    vlad_subtract_intranorm_kernel<<<blocks, threads>>>(
        vlad.data_ptr<float>(), a.data_ptr<float>(), out.data_ptr<float>(),
        B, D, K
    );
    return out;
}
"""

cpp_source = """
torch::Tensor softmax_slice_cuda(torch::Tensor input, int K);
torch::Tensor compute_a_cuda(torch::Tensor softmax_sliced, torch::Tensor clusters2, int B, int N, int D, int K);
torch::Tensor vlad_subtract_intranorm_cuda(torch::Tensor vlad, torch::Tensor a, int B, int D, int K);
"""

# Compile the inline CUDA code
custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["softmax_slice_cuda", "compute_a_cuda", "vlad_subtract_intranorm_cuda"],
    verbose=False,
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

        self.clusters = nn.Parameter(init_sc * torch.randn(feature_size, clusters))
        self.batch_norm = nn.BatchNorm1d(clusters)
        self.clusters2 = nn.Parameter(init_sc * torch.randn(1, feature_size, cluster_size))
        self.out_dim = self.cluster_size * feature_size

        self.custom_ops = custom_ops

    def forward(self, x, mask=None):
        max_sample = x.size(1)
        B = x.size(0)
        N = max_sample
        D = self.feature_size
        K = self.cluster_size
        G = self.ghost_clusters

        x = x.view(-1, D)  # (B*N, D)

        if x.device != self.clusters.device:
            msg = f"x.device {x.device} != cluster.device {self.clusters.device}"
            raise ValueError(msg)

        # Step 1: matmul + batch_norm (keep as PyTorch)
        assignment = torch.matmul(x, self.clusters)  # (BN, K+G)
        assignment = self.batch_norm(assignment)

        # Step 2: custom softmax + slice (fused)
        assignment = self.custom_ops.softmax_slice_cuda(assignment, K)  # (BN, K)

        # Step 3: reshape for later use
        assignment = assignment.view(B, N, K)  # (B, N, K)

        # Step 4: compute a = sum(assignment, dim=1) * clusters2 (fused)
        a = self.custom_ops.compute_a_cuda(assignment, self.clusters2, B, N, D, K)  # (B, D, K)

        # Step 5: prepare for second matmul
        assignment_t = assignment.transpose(1, 2).contiguous()  # (B, K, N)
        x = x.view(B, N, D)  # (B, N, D)

        # Step 6: second matmul (keep as PyTorch)
        vlad = torch.matmul(assignment_t, x)  # (B, K, D)
        vlad = vlad.transpose(1, 2).contiguous()  # (B, D, K)

        # Step 7: subtract a and intra-normalize (fused)
        vlad = self.custom_ops.vlad_subtract_intranorm_cuda(vlad, a, B, D, K)  # (B, D, K)

        # Step 8: flatten + global L2 norm (keep as PyTorch)
        vlad = vlad.reshape(B, K * D)
        vlad = F.normalize(vlad, p=2, dim=1)
        return vlad