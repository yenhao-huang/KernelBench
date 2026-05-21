import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from transformers import AutoModelForCausalLM, AutoConfig

# Define custom CUDA kernels for optimized GPT-2 operations
# We will replace:
# 1. The Embedding lookup (optimized with a simple kernel or just rely on PyTorch's highly optimized embedding)
#    Actually, for GPT-2, the bottleneck is usually the attention mechanism and MLP.
#    Let's focus on optimizing the Core Attention and MLP layers which are compute-heavy.

# However, writing a full fused attention + MLP kernel from scratch in inline CUDA is extremely complex 
# and error-prone without external libraries like FlashAttention or Triton.
# Given the constraints of "inline" and "real code", we will implement:
# 1. A custom Fused Linear Layer (Matmul + Bias) for the dense layers in MLP and Attention projections.
#    This reduces memory traffic by fusing the bias addition into the matmul kernel.
# 2. We will keep the rest of the architecture structure but replace specific linear operations.

# Note: Replacing every single operator is not feasible in a short inline block without breaking 
# the complex internal state of `AutoModelForCausalLM`. Instead, we create a custom implementation 
# of the core components that are bottlenecks.

# Custom CUDA Kernel for Fused Matmul + Bias (FP32)
fused_matmul_bias_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void fused_matmul_bias_kernel(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    const float* __restrict__ bias, 
    float* __restrict__ out, 
    int M, // Batch size * Sequence Length (or just Batch)
    int N, // Output features
    int K  // Input features
) {
    // Each thread block handles one row of the output matrix (one sample in batch)
    // We use a simple approach: each thread computes one element of the output for a specific row.
    // To optimize, we can use shared memory, but for inline simplicity and correctness, 
    // we'll do a standard tiled or direct computation if K is small, or just direct.
    // Given GPT-2 hidden_size=768, K=768, N=3072 (MLP) or 768 (Attn).
    
    int row = blockIdx.y * blockDim.y + threadIdx.y; // M dimension
    int col = blockIdx.x * blockDim.x + threadIdx.x; // N dimension

    if (row < M && col < N) {
        float sum = 0.0f;
        const float* A_row = A + row * K;
        const float* B_col = B + col * K; // B is stored as [K, N], so we access column-wise
        
        // Unrolling or simple loop for K
        for (int k = 0; k < K; ++k) {
            sum += A_row[k] * B_col[k];
        }
        
        out[row * N + col] = sum + bias[col];
    }
}

torch::Tensor fused_matmul_bias_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor bias) {
    // A: [M, K]
    // B: [K, N]
    // bias: [N]
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto out = torch::zeros({M, N}, A.options());

    const int block_size_x = 32;
    const int block_size_y = 8; // 32*8 = 256 threads per block
    
    dim3 block(block_size_x, block_size_y);
    dim3 grid((N + block_size_x - 1) / block_size_x, (M + block_size_y - 1) / block_size_y);

    fused_matmul_bias_kernel<<<grid, block>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        out.data_ptr<float>(), 
        M, N, K
    );

    return out;
}
"""

fused_matmul_bias_cpp_source = (
    "torch::Tensor fused_matmul_bias_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor bias);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_matmul_bias_cpp_source,
    cuda_sources=fused_matmul_bias_source,
    functions=["fused_matmul_bias_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class FusedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features).cuda()) # Note: B is [K, N] in kernel, so weight is [in, out] -> need transpose or adjust kernel. 
        # My kernel expects B as [K, N]. PyTorch Linear uses weight [out, in].
        # So we need to pass weight.t() to the kernel.
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features).cuda())
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        # x: [M, K]
        # weight: [out, in] -> we need [in, out] for the kernel
        w_t = self.weight.t() 
        if self.bias is not None:
            return fused_ops.fused_matmul_bias_cuda(x, w_t, self.bias)
        else:
            # Fallback to standard matmul if no bias (though GPT-2 uses bias)
            return torch.matmul(x, w_t)

# We will create a custom GPT2Model that uses our FusedLinear for the dense layers.
# The original AutoModelForCausalLM is complex. We will replicate the essential structure 
# of GPT-2 using PyTorch modules but replacing Linear with FusedLinear.

class CustomGPT2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = FusedLinear(config.n_embd, config.n_inner)
        self.c_proj = FusedLinear(config.n_inner, config.n_embd)
        self.act = nn.GELU()

    def forward(self, x):
        hidden_states = self.c_fc(x)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states

class CustomGPT2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = FusedLinear(config.n_embd, 3 * config.n_embd)
        self.c_proj = FusedLinear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(0.0)
        self.resid_dropout = nn.Dropout(0.0)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head

    def forward(self, x):
        B, T, C = x.size() # batch, time, channels
        
        # Query, Key, Value projection using fused linear
        qkv = self.c_attn(x) # [B, T, 3*C]
        
        # Split into Q, K, V
        q, k, v = qkv.split(self.n_embd, dim=2) # [B, T, C] each
        
        # Reshape for attention: [B, T, H, D] -> [B, H, T, D]
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Attention scores: [B, H, T, T]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # Softmax
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Weighted values: [B, H, T, D]
        y = att @ v 
        
        # Back to original shape: [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection using fused linear
        y = self.resid_dropout(self.c_proj(y))
        return y

class CustomGPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CustomGPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = CustomGPT2MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class ModelNew(torch.nn.Module):
    def __init__(self, model_name, config):
        super().__init__()
        self.model_name = model_name
        self.config = config
        
        # Embedding
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.n_embd)
        
        # Dropout
        self.drop = nn.Dropout(0.1)
        
        # Transformer Blocks
        self.h = nn.ModuleList([CustomGPT2Block(config) for _ in range(config.n_layer)])
        
        # Final LayerNorm
        self.ln_f = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        b, t = x.size()
        
        # Token embeddings
        tok_emb = self.wte(x)
        
        # Position embeddings
        positions = torch.arange(0, t, dtype=torch.long, device=x.device).unsqueeze(0)
        pos_emb = self.wpe(positions)
        
        x = self.drop(tok_emb + pos_emb)
        
        for block in self.h:
            x = block(x)
            
        x = self.ln_f(x)
        
        # LM Head (Linear projection to vocab size)
        logits = torch.matmul(x, self.wte.weight.t())
        
        return type('obj', (object,), {'logits': logits})()

import math

def get_inputs():
    inputs = torch.randint(0, config.vocab_size, (batch_size, sequence_length)).cuda()
    return [inputs]

def get_init_inputs():
    return [model_name, config]