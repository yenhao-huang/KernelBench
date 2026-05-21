import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from transformers import AutoModelForCausalLM, AutoConfig

# Define custom CUDA kernels for optimized operations
# We will replace the core Linear (Matmul) and Softmax operations which are often bottlenecks.
# Specifically, we implement a fused Matmul + Add Bias + ReLU (if applicable) or just high-performance Matmul.
# Given BART uses GELU, we will focus on optimizing the Linear layer (Matmul) and potentially the final projection.

optimized_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Helper for atomic add if needed, but here we use standard grid-stride loops for matmul-like ops or simple element-wise

// Kernel 1: Optimized Matrix Multiplication (GEMM) using shared memory tiling
// This is a simplified but effective GEMM kernel for FP32
__global__ void gemm_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float sA[16][16];
    __shared__ float sB[16][16];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * 16 + ty;
    int col = bx * 16 + tx;

    float sum = 0.0f;

    // Loop over K in tiles of 16
    for (int t = 0; t < (K + 15) / 16; ++t) {
        // Load A tile into shared memory
        if (row < M && (t * 16 + tx) < K) {
            sA[ty][tx] = A[row * K + t * 16 + tx];
        } else {
            sA[ty][tx] = 0.0f;
        }

        // Load B tile into shared memory
        if ((t * 16 + ty) < K && col < N) {
            sB[ty][tx] = B[(t * 16 + ty) * N + col];
        } else {
            sB[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        for (int k = 0; k < 16; ++k) {
            sum += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Kernel 2: Optimized Softmax with numerical stability
__global__ void softmax_kernel(const float* input, float* output, int rows, int cols) {
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row_idx < rows) {
        const float* row_ptr = input + row_idx * cols;
        float* out_row_ptr = output + row_idx * cols;

        // Find max for numerical stability
        float max_val = -INFINITY;
        for (int i = 0; i < cols; ++i) {
            if (row_ptr[i] > max_val) {
                max_val = row_ptr[i];
            }
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < cols; ++i) {
            float val = expf(row_ptr[i] - max_val);
            out_row_ptr[i] = val;
            sum += val;
        }

        // Normalize
        float inv_sum = 1.0f / sum;
        for (int i = 0; i < cols; ++i) {
            out_row_ptr[i] *= inv_sum;
        }
    }
}

// Kernel 3: Fused Linear Layer (Matmul + Bias Add)
// Assumes A is [M, K], B is [N, K] (transposed for coalesced access if needed, but here we assume standard layout)
// C = A * B^T + bias
__global__ void fused_linear_kernel(const float* A, const float* B, const float* bias, float* C, int M, int N, int K) {
    __shared__ float sA[16][16];
    __shared__ float sB[16][16];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * 16 + ty;
    int col = bx * 16 + tx;

    float sum = 0.0f;

    for (int t = 0; t < (K + 15) / 16; ++t) {
        if (row < M && (t * 16 + tx) < K) {
            sA[ty][tx] = A[row * K + t * 16 + tx];
        } else {
            sA[ty][tx] = 0.0f;
        }

        if ((t * 16 + ty) < K && col < N) {
            sB[ty][tx] = B[(t * 16 + ty) * N + col]; // Note: B is usually [N, K], so B[j*K+i] is element (j,i). Here we want B^T. 
            // Wait, standard matmul C = A @ B. If A is [M, K] and B is [K, N], then C[i,j] = sum_k A[i,k]*B[k,j].
            // In PyTorch Linear: out = input @ weight.T + bias. Weight is [out_features, in_features].
            // So if input is [M, K] (batch*seq, hidden), weight is [N, K]. 
            // We want C[M, N]. 
            // Let's stick to the GEMM logic: A[M, K], B_transposed[K, N] -> C[M, N].
            // But passing B as [N, K] means we need to access B[k, j] which is B[j*K + k].
        } else {
            sB[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < 16; ++k) {
             // A[row, t*16+k] * B[t*16+k, col]
             // sA[ty][k] is A[row, t*16+k]
             // We need B[t*16+k, col]. If B is stored as [N, K], then B[col * K + (t*16+k)]
             sum += sA[ty][k] * sB[k][tx]; 
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        float val = sum;
        if (bias != nullptr) {
            val += bias[col];
        }
        C[row * N + col] = val;
    }
}

// Corrected Fused Linear Kernel for PyTorch Linear semantics:
// Input: [M, K], Weight: [N, K], Bias: [N]
// Output: [M, N] where Out[i,j] = sum_k(Input[i,k] * Weight[j,k]) + Bias[j]
__global__ void fused_linear_kernel_corrected(const float* A, const float* B, const float* bias, float* C, int M, int N, int K) {
    __shared__ float sA[16][16];
    __shared__ float sB[16][16];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * 16 + ty; // Row in A (and C)
    int col = bx * 16 + tx; // Col in B_transposed (which corresponds to index in N for C)

    float sum = 0.0f;

    // We iterate over K dimension in tiles of 16
    for (int t = 0; t < (K + 15) / 16; ++t) {
        // Load A[row, t*16...t*16+15] into sA[ty][0..15]
        if (row < M && (t * 16 + tx) < K) {
            sA[ty][tx] = A[row * K + t * 16 + tx];
        } else {
            sA[ty][tx] = 0.0f;
        }

        // Load B[col, t*16...t*16+15] into sB[ty][0..15] 
        // Note: B is [N, K]. We want B[col, k]. So B[col * K + k].
        if ((t * 16 + ty) < K && col < N) {
            sB[ty][tx] = B[col * K + t * 16 + tx];
        } else {
            sB[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Dot product of row from A and col from B (which is a row in B since we loaded it directly)
        for (int k = 0; k < 16; ++k) {
            sum += sA[ty][k] * sB[ty][k]; // Wait, sB[ty][tx] was loaded with tx. 
            // Let's re-verify shared memory layout.
            // sA[ty][tx] holds A[row, t*16+tx]
            // sB[ty][tx] holds B[col, t*16+tx]
            // We want sum_k A[row, k] * B[col, k].
            // So we need to iterate k from 0 to 15 and multiply sA[ty][k] * sB[ty][k]? 
            // No, sA[ty][k] is A[row, t*16+k]. sB[ty][k] is B[col, t*16+k].
            // Yes, this works.
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        float val = sum;
        if (bias != nullptr) {
            val += bias[col];
        }
        C[row * N + col] = val;
    }
}


torch::Tensor fused_linear_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    // Input: [M, K], Weight: [N, K], Bias: [N]
    // Output: [M, N]
    
    int M = input.size(0);
    int K = input.size(1);
    int N = weight.size(0);

    auto output = torch::zeros({M, N}, input.options());

    if (M == 0 || N == 0 || K == 0) {
        return output;
    }

    const int block_size_x = 16;
    const int block_size_y = 16;
    
    dim3 block(block_size_x, block_size_y);
    dim3 grid((N + block_size_x - 1) / block_size_x, (M + block_size_y - 1) / block_size_y);

    fused_linear_kernel_corrected<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.numel() > 0 ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        M, N, K
    );

    return output;
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    // Input: [M, N]
    // Output: [M, N]
    
    int M = input.size(0);
    int N = input.size(1);

    auto output = torch::zeros_like(input);

    if (M == 0 || N == 0) {
        return output;
    }

    const int block_size = 256;
    dim3 block(block_size);
    dim3 grid((M + block_size - 1) / block_size);

    softmax_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N
    );

    return output;
}
"""

optimized_cpp_source = (
    "torch::Tensor fused_linear_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"
    "torch::Tensor softmax_cuda(torch::Tensor input);"
);

# Compile the inline CUDA code
custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=optimized_cpp_source,
    cuda_sources=optimized_cuda_source,
    functions=["fused_linear_cuda", "softmax_cuda"],
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
        
        # We will replace specific Linear layers with our custom fused linear implementation
        # BART's architecture consists of Encoder/Decoder blocks. 
        # The final logits projection is a Linear layer.
        # Also, internal FFN layers use Linear layers.
        
        # To make this robust and simple, we will wrap the original model's forward pass
        # but intercept calls to Linear layers if possible, or more practically,
        # since we can't easily monkey-patch every instance without modifying the source deeply,
        # we will create a new module that mimics the structure but uses our custom ops for the heavy lifting.
        
        # However, BART is complex. A simpler approach for "optimizing the architecture" in this context
        # is to replace the final projection and potentially key internal projections if we can access them.
        # But since `AutoModelForCausalLM` creates a fixed graph, let's try to replace the Linear layers 
        # by iterating through the model's children and replacing them.
        
        self._replace_linear_layers(self.original_model)
        
        # Store the modified model
        self.model = self.original_model

    def _replace_linear_layers(self, module):
        """Recursively replace nn.Linear with a custom wrapper that uses CUDA kernel"""
        for name, child in module.named_children():
            if isinstance(child, torch.nn.Linear):
                # Create a custom linear layer that uses our CUDA kernel
                setattr(module, name, CustomLinearWrapper(
                    child.in_features, 
                    child.out_features, 
                    child.bias is not None,
                    child.weight.data.clone(),
                    child.bias.data.clone() if child.bias is not None else None
                ))
            else:
                self._replace_linear_layers(child)

    def forward(self, x):
        return self.model(x).logits


class CustomLinearWrapper(torch.nn.Module):
    def __init__(self, in_features, out_features, has_bias, weight_data, bias_data=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Store weights and biases as buffers so they are part of the module state
        self.register_buffer('weight', weight_data)
        if has_bias:
            self.register_buffer('bias', bias_data)
        else:
            self.bias = None
            
    def forward(self, input):
        # Input shape: [M, in_features]
        # Weight shape: [out_features, in_features]
        
        # Use custom CUDA kernel for Matmul + Bias Add
        if self.bias is not None:
            out = custom_ops.fused_linear_cuda(input, self.weight, self.bias)
        else:
            # If no bias, we can still use the kernel but pass nullptr, or just matmul
            # Let's use a simple matmul kernel or reuse fused with null bias
            out = custom_ops.fused_linear_cuda(input, self.weight, torch.empty(0))
            
        return out


# Note: The above approach replaces ALL linear layers. 
# For BART, this might be overkill and memory intensive if not careful, but it fulfills the requirement.
# We also need to handle Softmax if present in the original model's forward pass for logits calculation.
# In CausalLM, the final step is often Linear -> LogSoftmax or similar.
# Let's check if we can intercept the final output.

# Actually, let's refine ModelNew to be more specific and efficient.
# We will replace the final projection layer specifically, as that's where logits are generated.
# And we'll assume the internal layers are fast enough or handled by cuDNN/cublas in the original model.
# But the prompt asks to optimize the architecture. Replacing all Linears is a valid optimization strategy.

# Let's adjust ModelNew to be cleaner and ensure it works with the provided inputs.

class ModelNew(torch.nn.Module):
    def __init__(self, model_name, config):
        super().__init__()
        self.model_name = model_name
        self.config = config
        
        # Load original model
        base_model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
        
        # Replace all Linear layers with our custom CUDA-accelerated wrappers
        self._replace_all_linears(base_model)
        
        self.model = base_model

    def _replace_all_linears(self, module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.Linear):
                weight_data = child.weight.data.clone()
                bias_data = child.bias.data.clone() if child.bias is not None else None
                
                setattr(module, name, CustomLinearWrapper(
                    child.in_features,
                    child.out_features,
                    child.bias is not None,
                    weight_data,
                    bias_data
                ))
            else:
                self._replace_all_linears(child)

    def forward(self, x):
        # The original model returns a ModelOutput with logits.
        # We want to return just the logits tensor as per the original architecture's output signature in the example?
        # Original: return self.model(x).logits
        return self.model(x).logits

# Re-defining CustomLinearWrapper to ensure it uses the compiled ops correctly
class CustomLinearWrapper(torch.nn.Module):
    def __init__(self, in_features, out_features, has_bias, weight_data, bias_data=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Use buffers to keep weights on GPU and part of the module state
        self.register_buffer('weight', weight_data)
        if has_bias:
            self.register_buffer('bias', bias_data)
        else:
            self.bias = None
            
    def forward(self, input):
        # Input: [M, in_features]
        # Weight: [out_features, in_features]
        
        # Call custom fused linear kernel
        if self.bias is not None:
            return custom_ops.fused_linear_cuda(input, self.weight, self.bias)
        else:
            # Pass empty tensor for bias to indicate no bias
            return custom_ops.fused_linear_cuda(input, self.weight, torch.empty(0))