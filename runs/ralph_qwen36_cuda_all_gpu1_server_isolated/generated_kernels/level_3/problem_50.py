import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernels for optimized operations
# 1. Custom Linear Layer (GEMM + Bias)
# 2. Custom Attention Kernel (Matmul, Scale, Mask, ReLU, Matmul) fused to reduce memory traffic

custom_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Helper for GELU if needed, but here we focus on the main bottleneck: Attention + Linear

// 1. Custom Linear Forward Kernel (Optimized GEMM with Bias)
__global__ void linear_forward_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ output, 
    int batch_size, 
    int seq_len, 
    int in_features, 
    int out_features) {
    
    // Each thread handles one element of the output matrix
    // Output shape: (batch_size * seq_len, out_features)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * out_features;

    if (idx < total_elements) {
        int sample_idx = idx / out_features;
        int feature_idx = idx % out_features;
        
        // Input row index for this sample
        int input_row = sample_idx * in_features;
        
        float sum = 0.0f;
        if (bias != nullptr) {
            sum = bias[feature_idx];
        }

        // Perform dot product
        const float* input_ptr = input + input_row;
        const float* weight_ptr = weight + feature_idx * in_features; // Weight is typically stored as [out, in] or [in, out]. 
        // PyTorch Linear: output = input @ weight.T + bias. 
        // So weight shape is (out_features, in_features).
        // We iterate over in_features.
        
        #pragma unroll
        for (int i = 0; i < in_features; ++i) {
            sum += input_ptr[i] * weight_ptr[i];
        }
        
        output[idx] = sum;
    }
}

// 2. Custom Attention Kernel
// Fuses: QK^T, Scale, Mask (Causal), ReLU, AV
__global__ void attention_kernel(
    const float* __restrict__ q,      // (B, nh, T, hs)
    const float* __restrict__ k,      // (B, nh, T, hs)
    const float* __restrict__ v,      // (B, nh, T, hs)
    float* __restrict__ output,       // (B, nh, T, hs)
    int B, 
    int nh, 
    int T, 
    int hs,
    float scale) {
    
    // Each block handles one head in one batch item? Or just one element?
    // To optimize, let's have each thread compute one output element (B, nh, t, h)
    // But we need to access the whole row of attention scores.
    // A better approach for fused attention: 
    // Each block processes one (b, head). The threads in the block process the sequence T.
    
    int b = blockIdx.z;
    int head = blockIdx.y;
    int t = threadIdx.x; // This thread computes output[t]
    
    if (t >= T) return;

    // Global indices
    int batch_head_idx = b * nh + head;
    int q_offset = batch_head_idx * T * hs;
    int k_offset = batch_head_idx * T * hs;
    int v_offset = batch_head_idx * T * hs;
    
    // We need to compute the attention score for query t against all keys 0..t
    // Then apply ReLU, then multiply by V.
    
    float sum_v = 0.0f;
    float max_score = -1e9; // For numerical stability if we were doing softmax, but here it's ReLU
    
    // Step 1: Compute scores and accumulate weighted values in one pass?
    // No, ReLU is non-linear. We must compute scores, apply mask/relu, then multiply by V.
    // However, we can fuse the accumulation of V if we store intermediate scores or recompute.
    // Given T=1024, storing 1024 floats in shared memory is feasible.
    
    // Let's use Shared Memory for the current row of Attention Scores (QK^T)
    extern __shared__ float s_score[];
    
    // Load Query vector t into registers/shared? 
    // Actually, let's load the whole Q[t] and K[0..t] into registers/shared.
    
    // Load Q[t]
    float q_vec[hs];
    for (int h = 0; h < hs; ++h) {
        q_vec[h] = q[q_offset + t * hs + h];
    }
    
    // We will compute the attention weights for key k_idx against query q_t
    // Then accumulate v[k_idx] * weight
    
    float acc_v[hs];
    for (int h = 0; h < hs; ++h) {
        acc_v[h] = 0.0f;
    }
    
    // Iterate over keys k_idx from 0 to t
    for (int k_idx = 0; k_idx <= t; ++k_idx) {
        float score = 0.0f;
        
        // Compute dot product Q[t] . K[k_idx]
        #pragma unroll
        for (int h = 0; h < hs; ++h) {
            score += q_vec[h] * k[k_offset + k_idx * hs + h];
        }
        
        score *= scale;
        
        // Apply Causal Mask: if k_idx > t, score is -inf. 
        // Our loop goes 0..t, so all are valid.
        // Apply ReLU
        if (score < 0.0f) {
            score = 0.0f;
        }
        
        // Accumulate V[k_idx] * score
        #pragma unroll
        for (int h = 0; h < hs; ++h) {
            acc_v[h] += v[v_offset + k_idx * hs + h] * score;
        }
    }
    
    // Write output
    int out_offset = batch_head_idx * T * hs;
    for (int h = 0; h < hs; ++h) {
        output[out_offset + t * hs + h] = acc_v[h];
    }
}

// Wrapper for Linear
torch::Tensor linear_forward_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto B = input.size(0);
    auto T = input.size(1);
    auto in_features = input.size(2);
    auto out_features = weight.size(0);
    
    auto output = torch::empty({B, T, out_features}, input.options());
    
    const int block_size = 256;
    int total_elements = B * T * out_features;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    // Note: This simple kernel is not highly optimized for GEMM. 
    // For production, one would use CUTLASS or cuBLAS. 
    // However, to demonstrate custom CUDA replacement as requested:
    
    linear_forward_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias != nullptr ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        B, T, in_features, out_features
    );
    
    return output;
}

// Wrapper for Attention
torch::Tensor attention_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    auto B = q.size(0);
    auto nh = q.size(1);
    auto T = q.size(2);
    auto hs = q.size(3);
    
    auto output = torch::empty_like(q);
    
    // Shared memory size: T * sizeof(float) for scores if we were doing softmax.
    // Here we don't store all scores, but let's allocate some shared mem for potential reuse or just rely on registers.
    // The kernel above doesn't use extern shared memory effectively for the whole row, 
    // but it avoids global memory reads for Q and K by keeping them in registers/shared per block.
    // Let's set shared memory to 0 for this simple register-based version to avoid allocation issues if T is large.
    // Actually, let's use shared memory to cache K rows if needed, but with hs=64, registers are fine.
    
    dim3 grid(B, nh, 1);
    int block_size = T; 
    // Cap block size at 1024
    if (block_size > 1024) block_size = 1024;
    
    float scale = 1.0f / sqrtf(hs);
    
    attention_kernel<<<grid, block_size, 0>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        output.data_ptr<float>(),
        B, nh, T, hs, scale
    );
    
    return output;
}

"""

custom_cpp_source = """
torch::Tensor linear_forward_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
torch::Tensor attention_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v);
"""

# Compile the inline CUDA code
cuda_module = load_inline(
    name="custom_ops",
    cpp_sources=custom_cpp_source,
    cuda_sources=custom_cuda_source,
    functions=["linear_forward_cuda", "attention_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"]
)


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super(NewGELU, self).__init__()
    
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class ModelNew(nn.Module):
    """
    A multi-head masked self-attention layer with a projection at the end that uses ReLU instead of Softmax.
    Optimized with custom CUDA operators for Linear and Attention.
    """

    def __init__(self, n_embd, n_head, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                                     .view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd
        
        # Bind custom CUDA functions
        self.linear_fwd = cuda_module.linear_forward_cuda
        self.attention_fwd = cuda_module.attention_cuda

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # Using custom linear kernel for c_attn
        qkv = self.linear_fwd(x, self.c_attn.weight, self.c_attn.bias)
        
        # Split into q, k, v
        # qkv shape: (B, T, 3*C)
        split_size = C
        q = qkv[:, :, :split_size]
        k = qkv[:, :, split_size:2*split_size]
        v = qkv[:, :, 2*split_size:]

        # Reshape and transpose for attention heads
        # (B, T, nh, hs) -> (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 

        # Custom fused attention kernel
        y = self.attention_fwd(q, k, v)

        # Reshape back: (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C) 

        # Output projection using custom linear kernel
        y = self.linear_fwd(y, self.c_proj.weight, self.c_proj.bias)

        return y

def get_inputs():
    # randomly generate input tensors based on the model architecture
    a = torch.rand(batch_size, max_seqlen, n_embd).cuda()
    return [a]

def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    return [n_embd, n_head, max_seqlen]