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
    } while(0)

// 1. Optimized LayerNorm Kernel
// Combines mean/variance calculation, normalization, and affine transformation into one kernel pass.
__global__ void layer_norm_kernel(const float* input, const float* weight, const float* bias, 
                                  float* output, int hidden_size, int batch_size) {
    extern __shared__ char shared_mem[];
    float* s_mean = (float*)shared_mem;
    float* s_var = s_mean + 1;

    int tid = threadIdx.x;
    int total_threads = blockDim.x;
    
    // Calculate mean
    float sum = 0.0f;
    for (int i = tid; i < hidden_size; i += total_threads) {
        sum += input[i];
    }
    s_mean[0] = sum;
    __syncthreads();

    // Parallel reduction for mean
    for (int stride = total_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_mean[0] += s_mean[stride];
        }
        __syncthreads();
    }
    
    float mean = s_mean[0] / hidden_size;

    // Calculate variance
    float var_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += total_threads) {
        float diff = input[i] - mean;
        var_sum += diff * diff;
    }
    s_var[0] = var_sum;
    __syncthreads();

    // Parallel reduction for variance
    for (int stride = total_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_var[0] += s_var[stride];
        }
        __syncthreads();
    }

    float var = s_var[0] / hidden_size;
    float inv_std = rsqrtf(var + 1e-5);

    // Apply normalization and affine transform
    for (int i = tid; i < hidden_size; i += total_threads) {
        float x_hat = (input[i] - mean) * inv_std;
        output[i] = weight[i] * x_hat + bias[i];
    }
}

torch::Tensor layer_norm_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    // Input shape: [batch_size, hidden_size]
    TORCH_CHECK(input.dim() == 2, "Input must be 2D tensor");
    TORCH_CHECK(weight.dim() == 1, "Weight must be 1D tensor");
    TORCH_CHECK(bias.dim() == 1, "Bias must be 1D tensor");
    
    int batch_size = input.size(0);
    int hidden_size = input.size(1);
    
    auto output = torch::empty_like(input);
    
    const int block_size = 256; // Must be power of 2 for reduction, or handle non-power-of-2 carefully. 
                                // For simplicity in this example, we assume hidden_size is manageable.
                                // If hidden_size < block_size, we still use block_size but only active threads do work.
    
    // Shared memory size: 2 floats (mean, var)
    const int shared_mem_size = 2 * sizeof(float);

    for (int b = 0; b < batch_size; ++b) {
        layer_norm_kernel<<<1, block_size, shared_mem_size>>>(
            input.data_ptr<float>() + b * hidden_size,
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            output.data_ptr<float>() + b * hidden_size,
            hidden_size,
            batch_size
        );
    }
    
    CUDA_CHECK(cudaGetLastError());
    return output;
}

// 2. Optimized Linear Kernel (Matmul)
// Uses a simple tiled approach or direct mapping for demonstration. 
// For real production, one would use CUTLASS or cuBLAS, but here we write inline to show the concept.
// We will use a straightforward row-major multiplication optimized for cache locality if possible,
// but given the constraints of inline code without external libraries like CUTLASS, 
// we'll implement a basic efficient version using shared memory tiling if dimensions allow, 
// or just a standard optimized loop structure.

__global__ void linear_kernel(const float* input, const float* weight, const float* bias, 
                              float* output, int in_features, int out_features) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < out_features && col < in_features) {
        // We compute the dot product for output[row]
        // However, standard grid mapping: each thread computes one element of output?
        // No, usually each thread computes one output element by iterating over input features.
        // Let's map: blockIdx.y -> output row (out_features), blockIdx.x -> block in input dimension?
        // Better: Each thread computes ONE output element.
    }
}

// Revised Linear Kernel: Each thread computes one output element
__global__ void linear_kernel_v2(const float* input, const float* weight, const float* bias, 
                                 float* output, int batch_size, int in_features, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size * out_features) {
        int b = idx / out_features;
        int o = idx % out_features;
        
        float sum = 0.0f;
        const float* input_row = input + b * in_features;
        const float* weight_col = weight + o * in_features; // Weight is [out_features, in_features]
        
        for (int i = 0; i < in_features; ++i) {
            sum += input_row[i] * weight_col[i];
        }
        
        output[idx] = sum + bias[o];
    }
}

torch::Tensor linear_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    // Input: [batch_size, in_features]
    // Weight: [out_features, in_features]
    // Bias: [out_features]
    
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(weight.dim() == 2, "Weight must be 2D");
    TORCH_CHECK(bias.dim() == 1, "Bias must be 1D");
    
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::empty({batch_size, out_features}, input.options());
    
    const int block_size = 256;
    int total_elements = batch_size * out_features;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    linear_kernel_v2<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );
    
    CUDA_CHECK(cudaGetLastError());
    return output;
}

"""

custom_cpp_source = """
torch::Tensor layer_norm_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
torch::Tensor linear_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Load the custom extensions
custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=custom_cpp_source,
    cuda_sources=custom_cuda_source,
    functions=["layer_norm_cuda", "linear_cuda"],
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
        
        # We need to reconstruct the model using our custom CUDA layers where possible.
        # BART's architecture consists of Embedding -> Encoder/Decoder Blocks (with LayerNorm and Linear/Matmul) -> Final Projection.
        # To keep it simple and functional without rewriting the entire transformer logic from scratch in Python/CUDA,
        # we will replace the specific linear layers and layer norms inside the model with our custom CUDA wrappers.
        
        self.model = base_model
        
        # Recursively replace Linear and LayerNorm modules with our custom CUDA implementations
        self._replace_modules(self.model)

    def _replace_modules(self, module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.Linear):
                # Replace standard Linear with our custom CUDA Linear
                setattr(module, name, CustomLinearWrapper(
                    child.in_features, 
                    child.out_features, 
                    child.bias is not None,
                    custom_ops.linear_cuda
                ))
            elif isinstance(child, torch.nn.LayerNorm):
                # Replace standard LayerNorm with our custom CUDA LayerNorm
                setattr(module, name, CustomLayerNormWrapper(
                    child.normalized_shape,
                    elementwise_affine=child.elementwise_affine,
                    eps=child.eps,
                    custom_ops.layer_norm_cuda
                ))
            else:
                self._replace_modules(child)

    def forward(self, x):
        return self.model(x).logits


class CustomLinearWrapper(torch.nn.Module):
    def __init__(self, in_features, out_features, has_bias, linear_func):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features))
        if has_bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.linear_func = linear_func

    def forward(self, input):
        # Ensure inputs are contiguous and float32 for CUDA kernel compatibility
        x = input.contiguous()
        w = self.weight.contiguous()
        b = self.bias.contiguous() if self.bias is not None else torch.zeros(0, dtype=x.dtype, device=x.device)
        
        return self.linear_func(x, w, b)


class CustomLayerNormWrapper(torch.nn.Module):
    def __init__(self, normalized_shape, elementwise_affine=True, eps=1e-5, layer_norm_func=None):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            self.weight = torch.nn.Parameter(torch.ones(*self.normalized_shape))
            self.bias = torch.nn.Parameter(torch.zeros(*self.normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            
        self.layer_norm_func = layer_norm_func

    def forward(self, input):
        # LayerNorm expects the last dimension to be normalized.
        # Our custom kernel assumes 2D input [batch, hidden]. 
        # If input is higher dimensional, we need to reshape or handle it.
        # BART's LayerNorms are typically on the last dim of [batch, seq, hidden] or [batch, hidden].
        
        original_shape = input.shape
        x = input.contiguous()
        
        # Reshape to 2D for the kernel: [batch * seq, hidden]
        if x.dim() > 2:
            x = x.view(-1, self.normalized_shape[-1])
            
        w = self.weight.contiguous() if self.weight is not None else torch.ones_like(x[:, :1])
        b = self.bias.contiguous() if self.bias is not None else torch.zeros_like(x[:, :1])
        
        out = self.layer_norm_func(x, w, b)
        
        # Reshape back to original shape
        out = out.view(original_shape)
        return out


# Initialize the optimized model
model_name = "facebook/bart-large"
config = AutoConfig.from_pretrained(model_name)

def get_inputs():
    inputs = torch.randint(0, config.vocab_size, (1, 1023))
    return [inputs]

def get_init_inputs():
    return [model_name, config]

# Instantiate the new model
ModelNew_instance = ModelNew(model_name, config)