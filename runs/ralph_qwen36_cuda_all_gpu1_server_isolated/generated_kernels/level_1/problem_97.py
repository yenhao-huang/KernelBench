import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA implementation of Scaled Dot-Product Attention
# This kernel fuses the steps: Q @ K^T, Softmax, and result @ V
# Optimized for FP32 precision.

sdpa_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Helper to get max of two floats
__device__ inline float fmaxf(float a, float b) {
    return (a > b) ? a : b;
}

// Kernel for Scaled Dot-Product Attention
// Computes: softmax((Q @ K^T) / sqrt(d_k)) @ V
// Q: [B, H, L_q, D]
// K: [B, H, L_k, D]
// V: [B, H, L_k, D]
// Out: [B, H, L_q, D]

__global__ void scaled_dot_product_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ Out,
    int B, int H, int L_q, int L_k, int D)
{
    // Each thread block handles one head and a chunk of queries or just one query?
    // For simplicity and correctness with shared memory constraints, 
    // we will assign one thread per output element (B, H, i, j).
    // This is memory bandwidth bound but simple to implement correctly without complex tiling.
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * H * L_q * L_k;

    if (idx >= total_elements) return;

    // Decode indices
    int temp = idx;
    int j = temp % L_k;       // Key index
    temp /= L_k;
    int i = temp % L_q;       // Query index
    temp /= L_q;
    int h = temp % H;         // Head index
    int b = temp / H;         // Batch index

    // Calculate base pointers for this specific (b, h) pair
    const float* q_ptr = Q + (b * H + h) * L_q * D + i * D;
    const float* k_ptr = K + (b * H + h) * L_k * D;
    const float* v_ptr = V + (b * H + h) * L_k * D;
    float* out_ptr = Out + (b * H + h) * L_q * D + i * D;

    // Step 1: Compute Q @ K^T for this query vector
    // We need to compute the dot product of q[i] with all k[j]
    // Then apply softmax
    
    float max_val = -INFINITY;
    float sum_exp = 0.0f;
    
    // First pass: find max and sum exp
    // To optimize, we can load K rows into registers or shared memory if D is small, 
    // but for general D=1024, global memory access is unavoidable per k row.
    // However, since we are doing one query at a time across all keys, 
    // we iterate through K.
    
    float scale = 1.0f / sqrtf((float)D);

    for (int d = 0; d < D; ++d) {
        // We need k[j][d] for all j. 
        // But wait, the loop above iterates j? No, the kernel is per (i,j).
        // Actually, standard softmax requires aggregating over ALL keys for a specific query i.
        // So we cannot compute the final output for (i, j) until we have processed all k's for that i.
        
        // Correction: The previous logic was flawed for a single-thread-per-output approach 
        // because Softmax is a reduction operation over the sequence dimension L_k.
        // We need to restructure the kernel or use atomic operations / shared memory.
    }

    // RESTRUCTURE: Use one thread block per Query (i) in a Head (h).
    // Block size = L_k (number of keys). 
    // Each thread computes one element of the attention score row for query i.
    // Then we do a reduction within the block to get softmax weights.
    
    // Let's restart the kernel logic with this structure.
}

// Better Kernel Structure:
// Grid: (B, H, L_q) -> 1D index mapped to blocks
// Block: (L_k) threads
// Each thread t in block computes score[i][t] = dot(Q[i], K[t])
// Then block reduction for softmax.
// Finally, each thread computes output vector as sum_t(softmax_score[t] * V[t])

__global__ void sdpa_kernel_v2(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ Out,
    int B, int H, int L_q, int L_k, int D)
{
    // Block index corresponds to a specific (b, h, i)
    int total_queries = B * H * L_q;
    int idx = blockIdx.x;
    
    if (idx >= total_queries) return;

    int b = idx / (H * L_q);
    int temp = idx % (H * L_q);
    int h = temp / L_q;
    int i = temp % L_q;

    // Shared memory for scores and partial sums if needed, but let's stick to registers/global for simplicity first.
    // Actually, for D=1024, loading K rows is expensive. 
    // We will use a simple approach: Each thread computes one attention score and accumulates into output.
    // But Softmax requires global knowledge of the row.
    
    // Approach: Two-pass or atomic accumulation?
    // Let's use a shared memory array for scores if L_k is small, but L_k=512 might be too big for shared mem per block if we store floats.
    // 512 * 4 bytes = 2KB. This fits easily in shared memory (usually 48KB+).
    
    extern __shared__ float sdata[]; 
    // Layout: sdata[threadIdx.x] stores the score for key threadIdx.x
    
    const float* q_row = Q + (b * H + h) * L_q * D + i * D;
    const float* k_base = K + (b * H + h) * L_k * D;
    const float* v_base = V + (b * H + h) * L_k * D;
    float* out_row = Out + (b * H + h) * L_q * D + i * D;

    int tid = threadIdx.x;
    
    // Step 1: Compute dot products Q[i] . K[t] for all t in this block
    float score = 0.0f;
    for (int d = 0; d < D; ++d) {
        score += q_row[d] * k_base[tid * D + d];
    }
    
    // Store score in shared memory
    sdata[tid] = score;
    __syncthreads();

    // Step 2: Find max of scores for softmax stability
    float max_score = -INFINITY;
    for (int t = 0; t < L_k; ++t) {
        if (sdata[t] > max_score) {
            max_score = sdata[t];
        }
    }
    
    // Step 3: Compute exp(score - max) and sum them up
    float sum_exp = 0.0f;
    for (int t = 0; t < L_k; ++t) {
        float val = expf(sdata[t] - max_score);
        sdata[tid] = val; // Overwrite with exp value
        sum_exp += val;
    }
    
    // We need the sum of all exp values. Since we overwrote sdata, we lost the original scores? 
    // No, we don't need original scores anymore, just the normalized weights.
    // But we need the total sum to normalize.
    // Let's use a simple reduction or atomic add for the sum if L_k is large, 
    // but since we have shared memory, we can do a tree reduction or just loop if L_k is small enough?
    // Looping for sum is O(N) per thread, which is fine.
    
    float total_sum = 0.0f;
    for (int t = 0; t < L_k; ++t) {
        total_sum += sdata[t];
    }
    
    float inv_sum = 1.0f / total_sum;

    // Step 4: Compute Output Vector
    // Out[i] = sum_t ( softmax_score[t] * V[t] )
    // softmax_score[t] = sdata[t] * inv_sum
    
    for (int d = 0; d < D; ++d) {
        float acc = 0.0f;
        for (int t = 0; t < L_k; ++t) {
            // Load V[t][d]
            acc += sdata[t] * inv_sum * v_base[t * D + d];
        }
        out_row[d] = acc;
    }
}

torch::Tensor sdpa_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
    
    auto B = Q.size(0);
    auto H = Q.size(1);
    auto L_q = Q.size(2);
    auto L_k = K.size(2);
    auto D = Q.size(3);

    TORCH_CHECK(K.size(3) == D, "K dimension mismatch");
    TORCH_CHECK(V.size(3) == D, "V dimension mismatch");
    TORCH_CHECK(L_q == Q.size(2), "Q sequence length mismatch");
    
    auto Out = torch::zeros_like(Q);

    const int block_size = L_k; // One thread per key
    // Ensure block size is valid (power of 2 is preferred for some optimizations, but not strictly required)
    // If L_k is very large, we might need to split, but 512 is fine.
    
    const int grid_size = B * H * L_q;
    
    // Shared memory size: L_k floats
    size_t shared_mem_bytes = block_size * sizeof(float);

    sdpa_kernel_v2<<<grid_size, block_size, shared_mem_bytes>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        Out.data_ptr<float>(),
        B, H, L_q, L_k, D
    );

    return Out;
}
"""

sdpa_cpp_source = (
    "torch::Tensor sdpa_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V);"
)

# Compile the inline CUDA code
sdpa_module = load_inline(
    name="sdpa_custom",
    cpp_sources=sdpa_cpp_source,
    cuda_sources=sdpa_source,
    functions=["sdpa_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # Use the custom CUDA operator
        return sdpa_module.sdpa_cuda(Q, K, V)