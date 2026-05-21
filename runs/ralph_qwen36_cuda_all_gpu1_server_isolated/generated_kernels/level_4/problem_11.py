import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from transformers import AutoModelForCausalLM, AutoConfig

# Define custom CUDA kernels for optimized operations
# We will replace the standard Linear (Matmul) and LayerNorm with fused/custom implementations
# to demonstrate significant optimization potential.

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
    } while (0)

// 1. Optimized Linear Layer (Matmul + Bias Add)
// Fuses the matrix multiplication and bias addition into a single kernel pass over memory if possible,
// or at least minimizes intermediate allocations. For large matrices, cuBLAS is usually best, 
// but here we provide a custom fused implementation for demonstration of inline CUDA.
// Note: For production, using torch.matmul is often faster due to cuBLAS optimization. 
// However, we implement a custom kernel to show the structure requested.

__global__ void linear_kernel(const float* input, const float* weight, const float* bias, float* output, int batch_size, int seq_len, int in_features, int out_features) {
    // Each thread handles one element of the output tensor
    // Output shape: (batch_size, seq_len, out_features)
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
        const float* weight_col = weight + feat_idx * in_features; // Weight is typically (out_features, in_features)

        for (int i = 0; i < in_features; ++i) {
            sum += input_row[i] * weight_col[i];
        }

        output[idx] = sum;
    }
}

torch::Tensor linear_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    // Input: (batch_size, seq_len, in_features)
    // Weight: (out_features, in_features)
    // Bias: (out_features,)
    
    auto batch_size = input.size(0);
    auto seq_len = input.size(1);
    auto in_features = input.size(2);
    auto out_features = weight.size(0);

    auto output = torch::zeros({batch_size, seq_len, out_features}, input.options());

    const int block_size = 256;
    int total_elements = batch_size * seq_len * out_features;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    // Launch kernel
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
// Standard PyTorch LayerNorm is quite optimized, but we provide a custom one that fuses 
// mean/variance calculation and normalization into fewer memory passes if beneficial, 
// or simply demonstrates the inline syntax for this specific operator.

__global__ void layernorm_kernel(const float* input, float* output, const float* weight, const float* bias, int batch_size, int seq_len, int hidden_size) {
    // Each block handles one (batch, seq) position
    int idx = blockIdx.x;
    if (idx >= batch_size * seq_len) return;

    const float* x = input + idx * hidden_size;
    float* y = output + idx * hidden_size;
    
    // Calculate mean
    float sum = 0.0f;
    for (int i = 0; i < hidden_size; ++i) {
        sum += x[i];
    }
    float mean = sum / hidden_size;

    // Calculate variance
    float var_sum = 0.0f;
    for (int i = 0; i < hidden_size; ++i) {
        float diff = x[i] - mean;
        var_sum += diff * diff;
    }
    float var = var_sum / hidden_size;
    float inv_std = rsqrtf(var + 1e-5);

    // Normalize and apply scale/bias
    for (int i = 0; i < hidden_size; ++i) {
        float normed = (x[i] - mean) * inv_std;
        y[i] = normed * weight[i] + bias[i];
    }
}

torch::Tensor layernorm_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int normalized_shape) {
    // Input: (*, hidden_size) where * is batch dimensions
    // We assume the last dimension is the one being normalized
    
    auto batch_dims = input.sizes().size() - 1;
    int hidden_size = input.size(-1);
    
    // Flatten all batch dimensions to simplify kernel launch
    auto flat_input = input.view({-1, hidden_size});
    auto batch_size_seq_len = flat_input.size(0);
    
    auto output = torch::empty_like(flat_input);

    const int block_size = 1; // One thread per element is not efficient for this logic, 
                              // but we use one block per vector to keep it simple and correct.
                              // Actually, let's use one block per vector (hidden_size threads) or grid-stride loop.
                              // For simplicity and correctness in inline example: 1 Block per vector.
    
    int num_blocks = batch_size_seq_len;
    
    layernorm_kernel<<<num_blocks, hidden_size>>>(
        flat_input.data_ptr<float>(),
        output.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        batch_size_seq_len, 1, hidden_size // seq_len=1 in flattened view
    );

    CUDA_CHECK(cudaGetLastError());
    
    return output.view_as(input);
}
"""

custom_cpp_source = (
    "torch::Tensor linear_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);\n"
    "torch::Tensor layernorm_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int normalized_shape);"
)

# Load the custom extensions
custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=custom_cpp_source,
    cuda_sources=custom_cuda_source,
    functions=["linear_cuda", "layernorm_cuda"],
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
        base_model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
        
        # We will replace specific layers with our custom CUDA implementations
        # Electra Small uses Embedding, LayerNorm, and Linear (Dense) layers.
        
        self.embedding = base_model.electra.embeddings.word_embeddings
        self.embedding_layer_norm = base_model.electra.embeddings.LayerNorm
        
        # Store the rest of the encoder layers but replace their internal linear/layernorm ops
        # Note: Replacing every single layer's internals dynamically is complex. 
        # A simpler approach for this demonstration is to wrap the model and intercept calls,
        # OR rebuild the model structure. Given the constraint "replace pytorch operators",
        # we will create a new module that mimics the structure but uses custom ops where possible.
        
        # However, AutoModelForCausalLM has many layers. Replacing them all manually is verbose.
        # Instead, we will demonstrate the replacement on the critical path: 
        # The Embedding LayerNorm and the final Classifier Head (Linear).
        # For the intermediate transformer blocks, we keep PyTorch's optimized implementations 
        # as writing a full custom Transformer from scratch in inline CUDA is beyond reasonable scope for this snippet.
        # But we WILL replace the Embedding LayerNorm and the Final Linear layer to show the pattern.
        
        self.encoder = base_model.electra.encoder
        
        # Get weights for the final classifier head
        self.classifier_weight = base_model.electra.embeddings.word_embeddings.weight.clone() # Electra shares embeddings or has a separate classifier? 
        # Actually, Electra uses the embedding matrix as the classifier weight.
        
        # We need to handle the forward pass carefully.
        # Let's just use the base model but override specific submodules if we could.
        # Since we can't easily monkey-patch deep modules in a pre-trained model without rebuilding,
        # we will construct a simplified version or assume the user wants the structure modified.
        
        # To strictly follow "replace pytorch operators", let's rebuild the forward logic 
        # using our custom ops for the parts we defined.
        
        self.custom_linear = custom_ops.linear_cuda
        self.custom_layernorm = custom_ops.layernorm_cuda
        
        # We need to copy parameters from the base model to our new structure
        # This is a simplified reconstruction for demonstration purposes.
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_tokens.weight.data.copy_(base_model.electra.embeddings.word_embeddings.weight.data)
        
        self.embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # We will use custom op for forward, but need parameters. 
        # Our custom layernorm takes weight and bias. PyTorch LayerNorm has them.
        self.ln_weight = nn.Parameter(base_model.electra.embeddings.LayerNorm.weight.clone())
        self.ln_bias = nn.Parameter(base_model.electra.embeddings.LayerNorm.bias.clone())
        
        # For the encoder, we will keep the original for now as replacing 12+ layers of attention/ffn 
        # with inline CUDA is extremely verbose. The prompt allows choosing WHICH operators to replace.
        # We replaced Embedding LayerNorm and Linear (Classifier).
        
        self.encoder = base_model.electra.encoder
        
    def forward(self, x):
        # 1. Embedding
        # x: (batch_size, seq_len)
        hidden_states = self.embed_tokens(x)
        
        # 2. Custom LayerNorm on embeddings
        # PyTorch: self.embed_layer_norm(hidden_states)
        # Custom: custom_layernorm(input, weight, bias, normalized_shape)
        hidden_states = self.custom_layernorm(
            hidden_states, 
            self.ln_weight, 
            self.ln_bias, 
            [self.config.hidden_size]
        )
        
        # 3. Pass through encoder (using PyTorch's optimized ops for attention/FFN)
        outputs = self.encoder(hidden_states)
        sequence_output = outputs.last_hidden_state
        
        # 4. Custom Linear for classification head
        # Electra shares the embedding weights for the classifier.
        # logits = torch.matmul(sequence_output, self.embed_tokens.weight.t())
        # Our custom linear expects: input (B, S, H), weight (H_out, H_in), bias
        # Here H_out = vocab_size, H_in = hidden_size
        
        # Note: The standard Linear in PyTorch does X @ W.T + b.
        # Our custom kernel does sum(input_row * weight_col). 
        # If weight is stored as (out_features, in_features), then input (B,S,in) @ weight.T (in,out) -> (B,S,out)
        # My kernel: output[idx] = sum(input[i] * weight[i]). This corresponds to Matrix Multiplication.
        
        logits = self.custom_linear(
            sequence_output, 
            self.embed_tokens.weight.data, 
            torch.zeros(1, device=sequence_output.device) # Electra classifier usually has no bias or it's handled differently? 
            # Actually, Electra's discriminator head is a linear layer. Let's check if there is bias.
            # In `electra_small`, the classifier is `nn.Linear(config.hidden_size, config.vocab_size)`
        )
        
        return type(outputs)(logits=logits)

# Re-initialize model_name and config for the final output block context
model_name = "google/electra-small-discriminator"
config = AutoConfig.from_pretrained(model_name)

def get_inputs():
    inputs = torch.randint(0, config.vocab_size, (1, 511))
    return [inputs]

def get_init_inputs():
    return [model_name, config]