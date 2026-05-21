import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from transformers import AutoModelForCausalLM, AutoConfig

# Define custom CUDA kernels for optimized GPT-2 operations
# We will replace the core Linear (Matmul) and LayerNorm operations with fused/custom implementations.
# Specifically, we implement a fused Matmul + Bias Add kernel and a standard LayerNorm kernel.
# Note: For GPT-2, the bottleneck is often the large matrix multiplications in the attention and MLP layers.

custom_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Kernel 1: Fused Matmul (A @ B^T) + Bias
// Assumes A is [M, K], B is [N, K] (transposed internally or passed as such), Out is [M, N]
// To simplify, we assume standard layout: A[M, K], B[N, K]. We compute C[i,j] = sum_k(A[i,k]*B[j,k]) + Bias[j]
__global__ void fused_matmul_bias_kernel(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    const float* __restrict__ Bias, 
    float* __restrict__ Out, 
    int M, int N, int K) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        const float* a_ptr = A + row * K;
        const float* b_ptr = B + col * K;
        
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            sum += a_ptr[k] * b_ptr[k];
        }
        
        if (Bias != nullptr) {
            sum += Bias[col];
        }
        Out[row * N + col] = sum;
    }
}

// Kernel 2: LayerNorm
// Input: [M, N], Weight: [N], Bias: [N], Output: [M, N]
__global__ void layer_norm_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ output, 
    int M, int N) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M) {
        // Calculate mean
        float sum = 0.0f;
        const float* input_row = input + row * N;
        for (int i = 0; i < N; ++i) {
            sum += input_row[i];
        }
        float mean = sum / N;

        // Calculate variance
        float var_sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            float diff = input_row[i] - mean;
            var_sum += diff * diff;
        }
        float var = var_sum / N + 1e-5; // epsilon
        float inv_std = rsqrtf(var);

        // Apply normalization and affine transform
        float* output_row = output + row * N;
        for (int i = 0; i < N; ++i) {
            float normalized = (input_row[i] - mean) * inv_std;
            if (weight != nullptr && bias != nullptr) {
                output_row[i] = weight[i] * normalized + bias[i];
            } else {
                output_row[i] = normalized;
            }
        }
    }
}

// Host functions to launch kernels
torch::Tensor fused_matmul_bias_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor Bias) {
    auto M = A.size(0);
    auto N = B.size(0);
    auto K = A.size(1);
    
    TORCH_CHECK(A.size(1) == B.size(1), "Input dimensions mismatch");
    
    auto out = torch::empty({M, N}, A.options());
    
    const int block_x = 32;
    const int block_y = 8;
    dim3 block(block_x, block_y);
    dim3 grid((N + block_x - 1) / block_x, (M + block_y - 1) / block_y);
    
    float* bias_ptr = Bias.numel() > 0 ? Bias.data_ptr<float>() : nullptr;
    
    fused_matmul_bias_kernel<<<grid, block>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        bias_ptr, 
        out.data_ptr<float>(), 
        M, N, K
    );
    
    return out;
}

torch::Tensor layer_norm_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto M = input.size(0);
    auto N = input.size(1);
    
    auto out = torch::empty_like(input);
    
    const int block_x = 32; // Not used directly in this simple grid but good practice
    const int block_y = 8;
    dim3 block(block_x, block_y);
    dim3 grid(1, (M + block_y - 1) / block_y);
    
    float* weight_ptr = weight.numel() > 0 ? weight.data_ptr<float>() : nullptr;
    float* bias_ptr = bias.numel() > 0 ? bias.data_ptr<float>() : nullptr;
    
    layer_norm_kernel<<<grid, block>>>(
        input.data_ptr<float>(), 
        weight_ptr, 
        bias_ptr, 
        out.data_ptr<float>(), 
        M, N
    );
    
    return out;
}
"""

custom_cpp_source = """
torch::Tensor fused_matmul_bias_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor Bias);
torch::Tensor layer_norm_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Load the custom extensions
cuda_module = load_inline(
    name="custom_gpt2_ops",
    cpp_sources=custom_cpp_source,
    cuda_sources=custom_cuda_source,
    functions=["fused_matmul_bias_cuda", "layer_norm_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"]
)

class ModelNew(torch.nn.Module):
    def __init__(self, model_name, config):
        super().__init__()
        self.model_name = model_name
        self.config = config
        
        # Load the original model to get weights and structure
        self.original_model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
        
        # We will replace specific layers with custom implementations.
        # GPT-2 consists of blocks containing:
        # 1. LayerNorm (ln_1)
        # 2. MultiHeadAttention (attn): 
        #    - q_proj, k_proj, v_proj (Linear)
        #    - attn_dropout
        #    - c_proj (Linear)
        # 3. Residual connection
        # 4. LayerNorm (ln_2)
        # 5. MLP (mlp):
        #    - c_fc (Linear) -> Gelu
        #    - c_proj (Linear)
        # 6. Residual connection
        
        # To fully replace the model with custom CUDA ops, we need to reconstruct the forward pass logic
        # because simply swapping layers in nn.Module is complex due to internal hooks and structure.
        # Instead, we will create a new module that mimics the GPT2Model structure but uses our custom kernels.
        
        self.vocab_size = config.vocab_size
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.n_layer = config.n_layer
        self.head_dim = self.n_embd // self.n_head
        
        # Copy weights from original model to new parameters
        # Embedding
        self.wte = nn.Parameter(self.original_model.transformer.wte.weight.clone())
        self.wpe = nn.Parameter(self.original_model.transformer.wpe.weight.clone())
        
        # Blocks
        self.blocks = nn.ModuleList()
        for i in range(self.n_layer):
            block = GPT2BlockCustom(
                n_embd=self.n_embd,
                n_head=self.n_head,
                head_dim=self.head_dim,
                ln_1_weight=self.original_model.transformer.h[i].ln_1.weight.clone(),
                ln_1_bias=self.original_model.transformer.h[i].ln_1.bias.clone(),
                attn_q_weight=self.original_model.transformer.h[i].attn.c_attn.weight[:self.n_embd, :].clone(), # c_attn is [3*emb, emb]
                attn_k_weight=self.original_model.transformer.h[i].attn.c_attn.weight[self.n_embd:2*self.n_embd, :].clone(),
                attn_v_weight=self.original_model.transformer.h[i].attn.c_attn.weight[2*self.n_embd:, :].clone(),
                attn_q_bias=self.original_model.transformer.h[i].attn.c_attn.bias[:self.n_embd].clone(),
                attn_k_bias=self.original_model.transformer.h[i].attn.c_attn.bias[self.n_embd:2*self.n_embd].clone(),
                attn_v_bias=self.original_model.transformer.h[i].attn.c_attn.bias[2*self.n_embd:].clone(),
                attn_out_weight=self.original_model.transformer.h[i].attn.c_proj.weight.clone(),
                attn_out_bias=self.original_model.transformer.h[i].attn.c_proj.bias.clone(),
                ln_2_weight=self.original_model.transformer.h[i].ln_2.weight.clone(),
                ln_2_bias=self.original_model.transformer.h[i].ln_2.bias.clone(),
                mlp_fc_weight=self.original_model.transformer.h[i].mlp.c_fc.weight.clone(),
                mlp_fc_bias=self.original_model.transformer.h[i].mlp.c_fc.bias.clone(),
                mlp_proj_weight=self.original_model.transformer.h[i].mlp.c_proj.weight.clone(),
                mlp_proj_bias=self.original_model.transformer.h[i].mlp.c_proj.bias.clone()
            )
            self.blocks.append(block)
            
        # Final LayerNorm
        self.ln_f_weight = self.original_model.transformer.ln_f.weight.clone()
        self.ln_f_bias = self.original_model.transformer.ln_f.bias.clone()

    def forward(self, x):
        # x: [batch_size, seq_len]
        batch_size, seq_len = x.size()
        
        # Get token and position embeddings
        tok_emb = self.wte(x)  # [B, S, D]
        pos_idx = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.wpe(pos_idx)  # [B, S, D]
        
        hidden_states = tok_emb + pos_emb
        
        for block in self.blocks:
            hidden_states = block(hidden_states)
            
        # Final LayerNorm
        hidden_states = cuda_module.layer_norm_cuda(hidden_states, self.ln_f_weight, self.ln_f_bias)
        
        # LM Head (Matmul with weight transpose)
        logits = cuda_module.fused_matmul_bias_cuda(hidden_states, self.wte.weight, torch.zeros(1, device=hidden_states.device))
        
        return type(self.original_model(x)).logits(logits)


class GPT2BlockCustom(nn.Module):
    def __init__(self, n_embd, n_head, head_dim, 
                 ln_1_weight, ln_1_bias,
                 attn_q_weight, attn_k_weight, attn_v_weight,
                 attn_q_bias, attn_k_bias, attn_v_bias,
                 attn_out_weight, attn_out_bias,
                 ln_2_weight, ln_2_bias,
                 mlp_fc_weight, mlp_fc_bias,
                 mlp_proj_weight, mlp_proj_bias):
        super().__init__()
        
        self.ln_1_weight = nn.Parameter(ln_1_weight)
        self.ln_1_bias = nn.Parameter(ln_1_bias)
        
        # Attention Weights
        self.attn_q_weight = nn.Parameter(attn_q_weight)
        self.attn_k_weight = nn.Parameter(attn_k_weight)
        self.attn_v_weight = nn.Parameter(attn_v_weight)
        self.attn_q_bias = nn.Parameter(attn_q_bias)
        self.attn_k_bias = nn.Parameter(attn_k_bias)
        self.attn_v_bias = nn.Parameter(attn_v_bias)
        
        self.attn_out_weight = nn.Parameter(attn_out_weight)
        self.attn_out_bias = nn.Parameter(attn_out_bias)
        
        self.ln_2_weight = nn.Parameter(ln_2_weight)
        self.ln_2_bias = nn.Parameter(ln_2_bias)
        
        # MLP Weights
        self.mlp_fc_weight = nn.Parameter(mlp_fc_weight)
        self.mlp_fc_bias = nn.Parameter(mlp_fc_bias)
        self.mlp_proj_weight = nn.Parameter(mlp_proj_weight)
        self.mlp_proj_bias = nn.Parameter(mlp_proj_bias)
        
        self.n_head = n_head
        self.head_dim = head_dim

    def forward(self, hidden_states):
        # 1. LayerNorm 1
        residual = hidden_states
        hidden_states = cuda_module.layer_norm_cuda(hidden_states, self.ln_1_weight, self.ln_1_bias)
        
        # 2. Attention
        # Q, K, V projections: [B, S, D] -> [B, S, D] for each
        q = cuda_module.fused_matmul_bias_cuda(hidden_states, self.attn_q_weight, self.attn_q_bias)
        k = cuda_module.fused_matmul_bias_cuda(hidden_states, self.attn_k_weight, self.attn_k_bias)
        v = cuda_module.fused_matmul_bias_cuda(hidden_states, self.attn_v_weight, self.attn_v_bias)
        
        # Reshape for attention: [B, S, D] -> [B, H, S, HeadDim]
        bsz, seq_len, _ = q.size()
        q = q.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        # Scores: [B, H, S, S]
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Softmax is usually done in FP32 for stability, but we can use PyTorch's optimized softmax
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Context: [B, H, S, HeadDim]
        context = torch.matmul(attn_weights, v)
        
        # Reshape back: [B, H, S, HeadDim] -> [B, S, D]
        context = context.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        
        # Output Projection
        attn_output = cuda_module.fused_matmul_bias_cuda(context, self.attn_out_weight, self.attn_out_bias)
        
        # Residual Connection 1
        hidden_states = attn_output + residual
        
        # 3. MLP
        residual = hidden_states
        hidden_states = cuda_module.layer_norm_cuda(hidden_states, self.ln_2_weight, self.ln_2_bias)
        
        # FC Layer
        fc_output = cuda_module.fused_matmul_bias_cuda(hidden_states, self.mlp_fc_weight, self.mlp_fc_bias)
        
        # Gelu Activation (using PyTorch's optimized version as it's hard to beat in pure CUDA for general cases without cuDNN)
        fc_output = torch.nn.functional.gelu(fc_output, approximate='tanh')
        
        # Projection Layer
        mlp_output = cuda_module.fused_matmul_bias_cuda(fc_output, self.mlp_proj_weight, self.mlp_proj_bias)
        
        # Residual Connection 2
        hidden_states = mlp_output + residual
        
        return hidden_states

model_name = "gpt2"
config = AutoConfig.from_pretrained(model_name)
vocab_size = config.vocab_size
sequence_length = 32
batch_size = 1024

def get_inputs():
    inputs = torch.randint(0, vocab_size, (batch_size, sequence_length))
    return [inputs]

def get_init_inputs():
    return [model_name, config]