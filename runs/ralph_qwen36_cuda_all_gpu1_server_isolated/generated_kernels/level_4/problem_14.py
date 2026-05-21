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

// 1. Optimized LayerNorm Kernel
// Combines mean/variance calculation, normalization, and affine transformation into one kernel pass.
__global__ void layer_norm_kernel(
    const float* input, 
    const float* weight, 
    const float* bias, 
    float* output, 
    int hidden_size, 
    int batch_size) 
{
    // Each block handles one sample in the batch
    int idx = blockIdx.x;
    if (idx >= batch_size) return;

    const float* x = input + idx * hidden_size;
    float* out = output + idx * hidden_size;
    
    // Shared memory for reduction is not strictly necessary for small hidden sizes, 
    // but global memory access pattern is critical. We use registers for accumulation.
    
    // Step 1: Calculate Mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        sum += x[i];
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

    // Step 2: Calculate Variance
    sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float diff = x[i] - mean;
        sum += diff * diff;
    }
    
    sdata[threadIdx.x] = sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float variance = sdata[0] / hidden_size;
    float rstd = rsqrtf(variance + 1e-5); // Epsilon for stability

    // Step 3: Normalize and Apply Affine Transformation
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float x_hat = (x[i] - mean) * rstd;
        out[i] = x_hat * weight[i] + bias[i];
    }
}

// 2. Optimized Matmul Kernel (GEMM)
// Uses a simple tiled approach for better cache utilization than naive implementation.
// For small matrices, this might not beat cuBLAS, but it demonstrates the structure.
// We assume A is [M, K] and B is [K, N], output C is [M, N].
__global__ void matmul_kernel(
    const float* A, 
    const float* B, 
    float* C, 
    int M, 
    int N, 
    int K) 
{
    // Tile size
    const int TILE_SIZE = 16;
    
    // Calculate global thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        
        // Loop over K dimension with tiling
        for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
            // Load elements into shared memory would be ideal, but for inline simplicity 
            // and avoiding complex synchronization logic in a single file without headers,
            // we do a direct computation. Note: In production, use cutlass or cuBLAS.
            // Here we just compute directly to ensure correctness and compilation ease.
            
            int k_idx = t * TILE_SIZE + threadIdx.x; // This is naive, but functional
            
            // Better approach for inline without shared memory complexity:
            // Just iterate K. For small K (e.g., 512), this is acceptable.
        }
        
        // Let's do a standard unrolled loop for correctness and simplicity in this context
        // Optimizing GEMM fully requires complex shared memory management which is verbose.
        // We will stick to a correct, reasonably optimized direct access pattern.
        
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        
        C[row * N + col] = sum;
    }
}

// Wrapper functions for PyTorch

torch::Tensor layer_norm_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto batch_size = input.size(0);
    auto hidden_size = input.size(1);
    
    auto output = torch::empty_like(input);
    
    const int block_size = 256; // Must be power of 2 for reduction logic
    dim3 threads(block_size);
    dim3 blocks(batch_size);
    
    // Shared memory size: one float per thread for reduction
    size_t shared_mem_size = block_size * sizeof(float);
    
    layer_norm_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        hidden_size,
        batch_size
    );
    
    CUDA_CHECK(cudaGetLastError());
    return output;
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    auto M = A.size(0);
    auto K = A.size(1);
    auto N = B.size(1);
    
    auto C = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    
    matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
    
    CUDA_CHECK(cudaGetLastError());
    return C;
}

"""

custom_cpp_source = """
torch::Tensor layer_norm_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

# Load the custom extensions
cuda_ops = load_inline(
    name="custom_ops",
    cpp_sources=custom_cpp_source,
    cuda_sources=custom_cuda_source,
    functions=["layer_norm_cuda", "matmul_cuda"],
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
        
        # We will replace specific layers with custom CUDA implementations
        # Electra Small uses Embedding -> LayerNorm -> TransformerBlock (with LayerNorm, Attention, FFN)
        
        # 1. Replace Embedding with a standard one (no easy CUDA speedup for lookup alone without batch optimization)
        self.embedding = self.original_model.electra.embeddings.word_embeddings
        
        # 2. Replace the initial LayerNorm
        self.initial_layer_norm = cuda_ops.layer_norm_cuda
        
        # We need to manually reconstruct the forward pass to inject custom ops
        # This is a simplified reconstruction for Electra-Small which has limited layers.
        # For a general solution, one would hook into the model's forward method or replace modules.
        # Here we explicitly define the forward logic using the custom ops where beneficial.
        
        # Store parameters from original model to use in custom ops
        self.register_buffer('ln_weight', self.original_model.electra.embeddings.LayerNorm.weight)
        self.register_buffer('ln_bias', self.original_model.electra.embeddings.LayerNorm.bias)
        
        # For the transformer blocks, we can replace the LayerNorms inside them.
        # However, replacing the entire attention/FFN block with a single CUDA kernel is complex 
        # and often not faster than cuBLAS/cuDNN for small models due to overhead.
        # The most impactful change here is fusing the Embedding + Initial LayerNorm or just optimizing LayerNorm.
        
        # Let's optimize the LayerNorms in the transformer blocks as well by replacing them.
        # We will iterate through encoder layers and replace their intermediate_layernorm and output_layernorm
        
        self.encoder = self.original_model.electra.encoder
        
        # Replace LayerNorm modules in encoder with custom CUDA function wrappers
        for i, layer in enumerate(self.encoder.layer):
            # Replace intermediate_layernorm
            layer.intermediate_layernorm = cuda_ops.layer_norm_cuda
            # Replace output_layernorm  
            layer.output_layernorm = cuda_ops.layer_norm_cuda
            
            # Note: The forward method of ElectraLayer calls these. 
            # We need to ensure the arguments match. 
            # Standard LayerNorm expects (input, weight, bias).
            # Our custom kernel expects (input, weight, bias).
            # However, PyTorch's nn.Module.forward signature is different from our function.
            # To make this work seamlessly without rewriting every layer's forward, 
            # we can create a wrapper module that mimics nn.LayerNorm but calls our CUDA op.
            
            class CustomLayerNormWrapper(nn.Module):
                def __init__(self, hidden_size, eps=1e-12):
                    super().__init__()
                    self.weight = nn.Parameter(torch.ones(hidden_size))
                    self.bias = nn.Parameter(torch.zeros(hidden_size))
                    self.eps = eps
                    # We don't use the CUDA op directly in forward because of signature mismatch
                    # Instead, we'll handle this by replacing the module with a functional call in a custom forward hook?
                    # No, simpler: Just replace the module with a custom class that implements __call__ correctly.
                
                def forward(self, x):
                    # PyTorch LayerNorm signature: forward(input)
                    # Our CUDA op: layer_norm_cuda(input, weight, bias)
                    return cuda_ops.layer_norm_cuda(x, self.weight, self.bias)

            # Re-assign the layers with wrappers that have the correct parameters from the original model
            # We need to copy weights from the original nn.LayerNorm to our wrapper
            orig_ln = layer.intermediate_layernorm
            new_ln = CustomLayerNormWrapper(orig_ln.normalized_shape[0], orig_ln.eps)
            new_ln.weight.data.copy_(orig_ln.weight.data)
            new_ln.bias.data.copy_(orig_ln.bias.data)
            layer.intermediate_layernorm = new_ln
            
            orig_ln_out = layer.output_layernorm
            new_ln_out = CustomLayerNormWrapper(orig_ln_out.normalized_shape[0], orig_ln_out.eps)
            new_ln_out.weight.data.copy_(orig_ln_out.weight.data)
            new_ln_out.bias.data.copy_(orig_ln_out.bias.data)
            layer.output_layernorm = new_ln_out

    def forward(self, x):
        # Embedding
        embedding_output = self.embedding(x)
        
        # Initial LayerNorm (Custom CUDA)
        # The original model applies LayerNorm after embedding
        # We need to get the weight and bias from the original structure or store them.
        # Since we replaced the encoder's LN, we still need to handle the initial one.
        # Let's use the stored buffers for the initial LN if it wasn't replaced, 
        # but actually, let's just replace the initial embedding layer norm too.
        
        # To keep it simple and robust, let's assume the initial LayerNorm is also handled by a wrapper
        # We didn't wrap the initial one in __init__, so we do it here or store it.
        # Let's create a wrapper for the initial LN as well.
        
        class InitialLayerNormWrapper(nn.Module):
            def __init__(self, weight, bias):
                super().__init__()
                self.weight = nn.Parameter(weight)
                self.bias = nn.Parameter(bias)
            
            def forward(self, x):
                return cuda_ops.layer_norm_cuda(x, self.weight, self.bias)

        # Get initial LN params from original model
        orig_init_ln = self.original_model.electra.embeddings.LayerNorm
        init_ln_wrapper = InitialLayerNormWrapper(orig_init_ln.weight.data, orig_init_ln.bias.data)
        
        embedding_output = init_ln_wrapper(embedding_output)
        
        # Pass through encoder (which now has custom LayerNorms in its layers)
        # We need to replicate the encoder's forward logic or call it.
        # Calling self.encoder.forward is tricky because it expects specific inputs.
        # ElectraEncoder.forward takes (hidden_states, attention_mask, head_mask, etc.)
        
        # Create a dummy attention mask for simplicity as in the original get_inputs
        batch_size = x.size(0)
        seq_len = x.size(1)
        attention_mask = torch.ones((batch_size, seq_len), device=x.device)
        
        # Call encoder forward. The internal layers now use our custom LayerNorm wrappers.
        encoder_outputs = self.encoder(hidden_states=embedding_output, attention_mask=attention_mask)
        
        hidden_states = encoder_outputs.last_hidden_state
        
        # Generator Head (Linear + Logits)
        # We can replace the final Linear layer with our custom matmul if desired, 
        # but for vocab_size x hidden_size, cuBLAS is usually very optimized.
        # However, let's demonstrate it.
        
        # The generator has a dense layer: linear(hidden_states, weight.T) + bias
        generator = self.original_model.electra.generator
        
        # Custom Matmul Wrapper
        class CustomLinearWrapper(nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(out_features, in_features))
                self.bias = nn.Parameter(torch.zeros(out_features))
            
            def forward(self, x):
                # x: [B, L, H] -> [B*L, H]
                # matmul expects [M, K] and [K, N]
                batch_seq_len = x.size(0) * x.size(1)
                hidden_size = x.size(2)
                
                x_flat = x.view(batch_seq_len, hidden_size)
                
                # Custom Matmul: A @ B.T ? 
                # Standard Linear is x @ W.T + b.
                # Our matmul does A @ B. So we need to pass W as B and transpose it?
                # Or just implement a linear kernel. Let's stick to the matmul kernel defined.
                # We need to transpose weight for the kernel: C = A @ B => [B*L, H] @ [H, V] -> [B*L, V]
                
                W_T = self.weight.t() # [H, V]
                
                logits_flat = cuda_ops.matmul_cuda(x_flat, W_T)
                
                # Add bias
                # Bias addition is element-wise, can be done in PyTorch or another kernel.
                # For simplicity, we do it in PyTorch as it's fast enough for the final step.
                logits_flat = logits_flat + self.bias
                
                return logits_flat.view(batch_seq_len, -1)

        # Replace generator dense layer with custom wrapper
        orig_dense = generator.dense
        custom_linear = CustomLinearWrapper(orig_dense.in_features, orig_dense.out_features)
        custom_linear.weight.data.copy_(orig_dense.weight.data)
        custom_linear.bias.data.copy_(orig_dense.bias.data)
        
        logits_flat = custom_linear(hidden_states)
        
        # Reshape back to [B, L, V]
        logits = logits_flat.view(batch_size, seq_len, -1)
        
        return type('Outputs', (), {'logits': logits})()

# Re-initialize model_name and config for the output context if needed, 
# but the class definition is self-contained.
model_name = "google/electra-small-discriminator"
config = AutoConfig.from_pretrained(model_name)

def get_inputs():
    inputs = torch.randint(0, config.vocab_size, (32, 256))
    return [inputs]

def get_init_inputs():
    return [model_name, config]