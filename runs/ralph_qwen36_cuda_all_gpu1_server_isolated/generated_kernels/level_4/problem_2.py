import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from transformers import AutoModelForCausalLM, AutoConfig

# Define custom CUDA kernels for optimized operations
# We will replace the standard Linear (Matmul) and LayerNorm with fused/custom implementations
# to reduce memory bandwidth overhead and improve compute efficiency.

custom_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s at line %d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// 1. Optimized Linear Layer (Matmul + Bias Add)
// Fuses the matrix multiplication and bias addition into a single kernel pass to save memory bandwidth.
__global__ void linear_kernel(const float* input, const float* weight, const float* bias, float* output, 
                              int batch_size, int seq_len, int in_features, int out_features) {
    // Each thread handles one element of the output tensor
    // Output shape: [batch_size, seq_len, out_features]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * out_features;

    if (idx < total_elements) {
        int sample_idx = idx / (seq_len * out_features);
        int token_idx = (idx % (seq_len * out_features)) / out_features;
        int feat_idx = idx % out_features;

        float sum = 0.0f;
        if (bias != nullptr) {
            sum = bias[feat_idx];
        }

        const float* input_row = input + sample_idx * seq_len * in_features + token_idx * in_features;
        const float* weight_col = weight + feat_idx * in_features; // Weight is typically [out_features, in_features]

        for (int i = 0; i < in_features; ++i) {
            sum += input_row[i] * weight_col[i];
        }

        output[idx] = sum;
    }
}

torch::Tensor linear_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    // Input: [batch_size, seq_len, in_features]
    // Weight: [out_features, in_features]
    // Bias: [out_features]
    
    auto batch_size = input.size(0);
    auto seq_len = input.size(1);
    auto in_features = input.size(2);
    auto out_features = weight.size(0);

    auto output = torch::empty({batch_size, seq_len, out_features}, input.options());

    const int block_size = 256;
    int total_elements = batch_size * seq_len * out_features;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    linear_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.numel() > 0 ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, seq_len, in_features, out_features
    );

    CUDA_CHECK(cudaGetLastError());
    return output;
}

// 2. Optimized LayerNorm
// Standard PyTorch LayerNorm involves multiple passes (mean, var, sub, div). 
// We fuse these into a single kernel for better performance on large sequences.
__global__ void layernorm_kernel(const float* input, const float* weight, const float* bias, float* output, 
                                 int batch_size, int seq_len, int hidden_size) {
    // Each block handles one token's normalization across the hidden dimension
    int token_idx = blockIdx.x;
    if (token_idx >= batch_size * seq_len) return;

    int base_idx = token_idx * hidden_size;
    
    // Step 1: Calculate Mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        sum += input[base_idx + i];
    }
    
    // Block-wide reduction for mean
    extern __shared__ float sdata[];
    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float mean = sdata[0] / hidden_size;

    // Step 2: Calculate Variance and Normalize
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float diff = input[base_idx + i] - mean;
        var_sum += diff * diff;
    }

    // Block-wide reduction for variance
    sdata[threadIdx.x] = var_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float variance = sdata[0] / hidden_size;
    float rstd = 1.0f / sqrtf(variance + 1e-5); // eps = 1e-5

    // Step 3: Apply Weight, Bias and Store
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float normalized = (input[base_idx + i] - mean) * rstd;
        if (weight != nullptr) {
            normalized = normalized * weight[i];
        }
        if (bias != nullptr) {
            normalized = normalized + bias[i];
        }
        output[base_idx + i] = normalized;
    }
}

torch::Tensor layernorm_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int normalized_shape) {
    // Input: [batch_size, seq_len, hidden_size]
    // Weight/Bias: [hidden_size]
    
    auto batch_size = input.size(0);
    auto seq_len = input.size(1);
    auto hidden_size = input.size(2);

    auto output = torch::empty_like(input);

    const int block_size = 256; // Must be power of 2 for reduction
    // Ensure block size is not larger than hidden_size to avoid issues, though typically hidden_size > 256
    if (block_size > hidden_size) {
        // Fallback or adjust logic, but for OPT-1.3B hidden_size=2048, 256 is fine.
    }

    int num_blocks = batch_size * seq_len;
    
    // Shared memory size: block_size floats for reduction
    size_t shared_mem_size = block_size * sizeof(float);

    layernorm_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.numel() > 0 ? weight.data_ptr<float>() : nullptr,
        bias.numel() > 0 ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, seq_len, hidden_size
    );

    CUDA_CHECK(cudaGetLastError());
    return output;
}

"""

custom_cpp_source = """
torch::Tensor linear_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
torch::Tensor layernorm_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int normalized_shape);
"""

# Load the custom extensions
cuda_module = load_inline(
    name="custom_ops",
    cpp_sources=custom_cpp_source,
    cuda_sources=custom_cuda_source,
    functions=["linear_cuda", "layernorm_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"]
)


class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # x: [batch, seq, in_features]
        return cuda_module.linear_cuda(x, self.weight, self.bias)


class CustomLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(*self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(*self.normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # x: [batch, seq, hidden]
        # We assume the last dimension is the one being normalized based on OPT architecture
        if len(self.normalized_shape) == 1:
            hidden_size = self.normalized_shape[0]
        else:
            # For multi-dim, we flatten everything except the last dim? 
            # Standard LayerNorm in transformers usually normalizes over the last dim.
            # Let's assume standard usage where normalized_shape matches the last dim(s).
            # For OPT, it's just the last dimension (hidden_size).
            hidden_size = x.size(-1)
        
        return cuda_module.layernorm_cuda(x, self.weight, self.bias, hidden_size)


import math

class ModelNew(torch.nn.Module):
    def __init__(self, model_name, config):
        super().__init__()
        self.model_name = model_name
        self.config = config
        
        # Load the base model to get weights
        base_model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
        
        # We will reconstruct the model using our custom layers while copying weights
        # This is a simplified reconstruction. A full replacement would require mapping every layer.
        # For OPT-1.3B, the main bottlenecks are the Linear layers in the decoder and the final LM head.
        # We will replace the core components.
        
        self.embed_tokens = base_model.model.decoder.embed_tokens
        self.final_layer_norm = CustomLayerNorm(
            config.hidden_size, 
            eps=config.layer_norm_eps
        )
        self.final_layer_norm.weight.data.copy_(base_model.model.decoder.layer_norm.weight.data)
        if hasattr(base_model.model.decoder.layer_norm, 'bias'):
             self.final_layer_norm.bias.data.copy_(base_model.model.decoder.layer_norm.bias.data)

        # Replace decoder layers
        self.decoder_layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            self.decoder_layers.append(CustomDecoderLayer(base_model.model.decoder.layers[i], config))

        self.lm_head = CustomLinear(
            config.hidden_size, 
            config.vocab_size, 
            bias=False
        )
        # Tie weights: lm_head weight is same as embed_tokens
        self.lm_head.weight.data.copy_(self.embed_tokens.weight.data)

    def forward(self, x):
        # x: [batch, seq]
        hidden_states = self.embed_tokens(x)
        
        for layer in self.decoder_layers:
            hidden_states = layer(hidden_states)
            
        hidden_states = self.final_layer_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return type('obj', (object,), {'logits': logits})()


class CustomDecoderLayer(nn.Module):
    def __init__(self, original_layer, config):
        super().__init__()
        self.self_attn = CustomAttention(original_layer.self_attn, config)
        self.self_attn_layer_norm = CustomLayerNorm(
            config.hidden_size, 
            eps=config.layer_norm_eps
        )
        self.self_attn_layer_norm.weight.data.copy_(original_layer.self_attn_layer_norm.weight.data)
        if hasattr(original_layer.self_attn_layer_norm, 'bias'):
             self.self_attn_layer_norm.bias.data.copy_(original_layer.self_attn_layer_norm.bias.data)

        self.fc1 = CustomLinear(
            config.hidden_size, 
            config.ffn_dim, 
            bias=True
        )
        self.fc1.weight.data.copy_(original_layer.fc1.weight.data)
        self.fc1.bias.data.copy_(original_layer.fc1.bias.data)

        self.fc2 = CustomLinear(
            config.ffn_dim, 
            config.hidden_size, 
            bias=True
        )
        self.fc2.weight.data.copy_(original_layer.fc2.weight.data)
        self.fc2.bias.data.copy_(original_layer.fc2.bias.data)

        self.final_layer_norm = CustomLayerNorm(
            config.hidden_size, 
            eps=config.layer_norm_eps
        )
        self.final_layer_norm.weight.data.copy_(original_layer.final_layer_norm.weight.data)
        if hasattr(original_layer.final_layer_norm, 'bias'):
             self.final_layer_norm.bias.data.copy_(original_layer.final_layer_norm.bias.data)

    def forward(self, x):
        # Self Attention
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x)
        x = x + residual

        # MLP
        residual = x
        x = self.final_layer_norm(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x + residual
        
        return x


class CustomAttention(nn.Module):
    def __init__(self, original_attn, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = float(self.head_dim) ** -0.5
        
        # Q, K, V projections
        self.q_proj = CustomLinear(
            self.embed_dim, 
            self.embed_dim, 
            bias=True
        )
        self.q_proj.weight.data.copy_(original_attn.q_proj.weight.data)
        self.q_proj.bias.data.copy_(original_attn.q_proj.bias.data)

        self.k_proj = CustomLinear(
            self.embed_dim, 
            self.embed_dim, 
            bias=True
        )
        self.k_proj.weight.data.copy_(original_attn.k_proj.weight.data)
        self.k_proj.bias.data.copy_(original_attn.k_proj.bias.data)

        self.v_proj = CustomLinear(
            self.embed_dim, 
            self.embed_dim, 
            bias=True
        )
        self.v_proj.weight.data.copy_(original_attn.v_proj.weight.data)
        self.v_proj.bias.data.copy_(original_attn.v_proj.bias.data)

        # Output projection
        self.out_proj = CustomLinear(
            self.embed_dim, 
            self.embed_dim, 
            bias=True
        )
        self.out_proj.weight.data.copy_(original_attn.out_proj.weight.data)
        self.out_proj.bias.data.copy_(original_attn.out_proj.bias.data)

    def forward(self, x):
        # x: [batch, seq, embed_dim]
        bsz, seq_len, _ = x.size()
        
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot Product Attention
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        
        # Causal Mask
        mask = torch.triu(torch.ones_like(attn_weights), diagonal=1).bool()
        attn_weights = attn_weights.masked_fill(mask, float('-inf'))
        
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_probs, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)
        
        return self.out_proj(attn_output)


def get_inputs():
    inputs = torch.randint(0, vocab_size, (batch_size, sequence_length))
    return [inputs]

def get_init_inputs():
    return [model_name, config]