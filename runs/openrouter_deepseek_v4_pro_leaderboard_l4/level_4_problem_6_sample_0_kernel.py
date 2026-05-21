import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.models.bart.modeling_bart import BartAttention

# CUDA kernel for fused scaled dot-product attention with causal masking
fused_attention_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_Q 32
#define TILE_K 32
#define HEAD_DIM 64

__global__ void fused_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const float scale
) {
    // Block index: blockIdx.x = batch*num_heads index, blockIdx.y = query tile index
    int b = blockIdx.x / num_heads;
    int h = blockIdx.x % num_heads;
    int q_tile_idx = blockIdx.y;
    int q_start = q_tile_idx * TILE_Q;
    int q_end = min(q_start + TILE_Q, seq_len);
    int q_tile_size = q_end - q_start;

    // Base pointers for this head
    const float* Q_head = Q + (b * num_heads + h) * seq_len * HEAD_DIM;
    const float* K_head = K + (b * num_heads + h) * seq_len * HEAD_DIM;
    const float* V_head = V + (b * num_heads + h) * seq_len * HEAD_DIM;
    float* O_head = O + (b * num_heads + h) * seq_len * HEAD_DIM;

    // Shared memory for K and V tiles
    __shared__ float K_tile[TILE_K][HEAD_DIM];
    __shared__ float V_tile[TILE_K][HEAD_DIM];

    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Registers for online softmax state
    float m[TILE_Q];   // running max
    float l[TILE_Q];   // running sum
    float O_reg[TILE_Q][HEAD_DIM]; // output accumulator

    // Initialize for each query in tile
    for (int i = 0; i < q_tile_size; i++) {
        m[i] = -INFINITY;
        l[i] = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) {
            O_reg[i][d] = 0.0f;
        }
    }

    // Loop over key tiles
    for (int k_start = 0; k_start < seq_len; k_start += TILE_K) {
        int k_end = min(k_start + TILE_K, seq_len);
        int k_tile_size = k_end - k_start;

        // Load K and V tiles into shared memory
        for (int i = ty; i < k_tile_size; i += blockDim.y) {
            for (int d = tx; d < HEAD_DIM; d += blockDim.x) {
                K_tile[i][d] = K_head[(k_start + i) * HEAD_DIM + d];
                V_tile[i][d] = V_head[(k_start + i) * HEAD_DIM + d];
            }
        }
        __syncthreads();

        // Compute attention scores for this tile
        for (int i = 0; i < q_tile_size; i++) {
            int q_idx = q_start + i;
            float row_max = -INFINITY;
            float scores[TILE_K];

            // Compute dot products and apply causal mask
            for (int j = 0; j < k_tile_size; j++) {
                int k_idx = k_start + j;
                if (k_idx > q_idx) {
                    scores[j] = -INFINITY; // causal mask
                } else {
                    float dot = 0.0f;
                    for (int d = 0; d < HEAD_DIM; d++) {
                        dot += Q_head[q_idx * HEAD_DIM + d] * K_tile[j][d];
                    }
                    scores[j] = dot * scale;
                }
                row_max = fmaxf(row_max, scores[j]);
            }

            // Update running max and sum
            float m_new = fmaxf(m[i], row_max);
            float l_new = expf(m[i] - m_new) * l[i];
            for (int j = 0; j < k_tile_size; j++) {
                if (scores[j] != -INFINITY) {
                    l_new += expf(scores[j] - m_new);
                }
            }

            // Update output accumulator
            float exp_scale = expf(m[i] - m_new);
            for (int d = 0; d < HEAD_DIM; d++) {
                O_reg[i][d] *= exp_scale;
            }
            for (int j = 0; j < k_tile_size; j++) {
                if (scores[j] != -INFINITY) {
                    float weight = expf(scores[j] - m_new);
                    for (int d = 0; d < HEAD_DIM; d++) {
                        O_reg[i][d] += weight * V_tile[j][d];
                    }
                }
            }

            m[i] = m_new;
            l[i] = l_new;
        }
        __syncthreads();
    }

    // Finalize and write output
    for (int i = 0; i < q_tile_size; i++) {
        int q_idx = q_start + i;
        float inv_l = 1.0f / l[i];
        for (int d = 0; d < HEAD_DIM; d++) {
            O_head[q_idx * HEAD_DIM + d] = O_reg[i][d] * inv_l;
        }
    }
}

torch::Tensor fused_attention_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float scale
) {
    const int batch_size = Q.size(0);
    const int num_heads = Q.size(1);
    const int seq_len = Q.size(2);
    const int head_dim = Q.size(3);

    auto O = torch::zeros_like(Q);

    const int q_tiles = (seq_len + TILE_Q - 1) / TILE_Q;
    const dim3 grid(batch_size * num_heads, q_tiles);
    const dim3 block(HEAD_DIM, TILE_K); // threads: x for head_dim, y for loading K/V

    fused_attention_kernel<<<grid, block>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        batch_size,
        num_heads,
        seq_len,
        scale
    );

    return O;
}
"""

fused_attention_cpp_source = (
    "torch::Tensor fused_attention_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V, float scale);"
)

# Compile the inline CUDA code
fused_attention = load_inline(
    name="fused_attention",
    cpp_sources=fused_attention_cpp_source,
    cuda_sources=fused_attention_source,
    functions=["fused_attention_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class FusedBartAttention(nn.Module):
    """Custom attention module that uses fused CUDA kernel for self-attention."""
    def __init__(self, original_attn):
        super().__init__()
        # Copy parameters from original BartAttention
        self.q_proj = original_attn.q_proj
        self.k_proj = original_attn.k_proj
        self.v_proj = original_attn.v_proj
        self.out_proj = original_attn.out_proj
        self.num_heads = original_attn.num_heads
        self.head_dim = original_attn.head_dim
        self.scaling = original_attn.scaling
        self.original_forward = original_attn.forward  # fallback for cross-attention

    def forward(
        self,
        hidden_states,
        key_value_states=None,
        past_key_value=None,
        attention_mask=None,
        layer_head_mask=None,
        output_attentions=False,
    ):
        # If cross-attention (key_value_states is not None), fall back to original
        if key_value_states is not None:
            return self.original_forward(
                hidden_states,
                key_value_states,
                past_key_value,
                attention_mask,
                layer_head_mask,
                output_attentions,
            )

        # Self-attention with causal mask (assumed for decoder)
        batch_size, seq_len, embed_dim = hidden_states.size()
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Reshape to (batch, num_heads, seq_len, head_dim)
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply fused attention
        scale = self.scaling
        attn_output = fused_attention.fused_attention_cuda(query, key, value, scale)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        attn_output = self.out_proj(attn_output)

        return (attn_output, None)  # match BartAttention output format


class ModelNew(nn.Module):
    def __init__(self, model_name, config):
        super().__init__()
        self.model_name = model_name
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(model_name, config=config)

        # Replace all BartAttention modules with FusedBartAttention
        self._replace_attention_modules(self.model)

    def _replace_attention_modules(self, module):
        for name, child in module.named_children():
            if isinstance(child, BartAttention):
                setattr(module, name, FusedBartAttention(child))
            else:
                self._replace_attention_modules(child)

    def forward(self, x):
        return self.model(x).logits