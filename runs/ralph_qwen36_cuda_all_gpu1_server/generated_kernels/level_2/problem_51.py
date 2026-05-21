import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels for the optimized operations.
# We will fuse: Gemm + Subtract -> GlobalAvgPool -> LogSumExp -> GELU -> ResidualAdd
# However, since ResidualAdd involves the original input 'x' which is not available in the intermediate stream easily without storing it,
# and GlobalAvgPool reduces dimensions, we need to be careful.
# The residual connection adds the original (batch_size, in_features) to the result of GELU(LogSumExp(GlobalAvgPool(Subtract(Gemm(x))))).
# Wait, looking at the shapes:
# x: (B, in_features) -> Gemm -> (B, out_features)
# Subtract -> (B, out_features)
# GlobalAvgPool(dim=1) -> (B, 1)
# LogSumExp(dim=1) -> (B, 1)
# GELU -> (B, 1)
# ResidualAdd: x + original_x. 
# original_x is (B, in_features).
# The current x is (B, 1).
# This addition is invalid unless broadcasting or if the problem implies a different structure.
# Let's re-read carefully: "ResidualAdd: x + original_x".
# If x is (B, 1) and original_x is (B, in_features), PyTorch will broadcast. 
# But usually residual connections are element-wise on same shapes.
# Let's assume the standard broadcasting behavior of PyTorch where (B, 1) + (B, in_features) results in (B, in_features).

# To optimize this effectively with CUDA, we can create a single kernel that performs:
# 1. Matrix Multiplication (Gemm)
# 2. Subtraction of bias vector
# 3. Global Average Pooling (sum and divide by out_features)
# 4. LogSumExp
# 5. GELU
# 6. Broadcast add with original input

# However, implementing a full fused kernel for such disparate operations in one go is complex and error-prone in inline CUDA.
# A more robust approach for "speedups" in this specific chain is to optimize the heavy parts: Gemm and the reduction/activation chain.
# But since we want to replace operators, let's look at what can be fused.
# The sequence after Gemm is: Subtract -> Mean -> LogSumExp -> GELU -> Add(original_x).
# This entire tail can be a single kernel if we pass original_x and the subtract parameter.

# Let's define two kernels for clarity and stability:
# 1. A fused kernel for the "Head" part: Gemm + Subtract + GlobalAvgPool + LogSumExp + GELU -> Output (B, 1)
#    Actually, we need original_x for the final add. So let's make a kernel that takes x, W, b_sub, and original_x, 
#    computes the head, and adds original_x.

# Kernel: Fused Head + Residual Add
# Inputs: 
#   x: (B, in_features)
#   W: (out_features, in_features) - Note: nn.Linear uses (out, in) for weight matrix if we do x @ W.T or W @ x?
#   PyTorch Linear: out = x @ W.T + b. So W is (out_features, in_features).
#   b_sub: (out_features,)
#   original_x: (B, in_features)
# Output: (B, in_features)

# We will use cuBLAS for the Gemm part as it's highly optimized, and then a custom kernel for the rest.
# Or we can write a pure CUDA kernel for everything if B is small, but for B=2048, cuBLAS is better for Gemm.
# The prompt asks to replace PyTorch operators with custom CUDA operators. 
# Using load_inline allows us to mix C++ and CUDA. We can call cuBLAS from the C++ wrapper.

optimized_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

// Helper for GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
__device__ __forceinline__ float gelu(float x) {
    float cdf = 0.5f * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
    return x * cdf;
}

// Kernel for the tail operations: Subtract, GlobalAvgPool, LogSumExp, GELU, and Residual Add
// This kernel assumes the Gemm has already been performed or we do it here.
// To maximize speedup, let's do the whole thing in one pass if possible, but Gemm is heavy.
// Let's split: 
// 1. Custom Gemm + Subtract + Reduction + Activation + Residual Add in a single kernel? 
//    This might be slow due to memory bandwidth limits for large matrices.
// 2. Better: Use cuBLAS for Gemm, then a custom kernel for the rest.

// Kernel: Tail operations
// Input: gemm_out (B, out_features), b_sub (out_features), original_x (B, in_features)
// Output: result (B, in_features)
__global__ void tail_kernel(const float* gemm_out, const float* b_sub, const float* original_x, 
                            float* result, int batch_size, int out_features, int in_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    // Each thread handles one sample in the batch? No, we need to reduce over out_features.
    // Let's have each thread block handle one sample.
    extern __shared__ float shared_mem[];
    
    int tid = threadIdx.x;
    int b = blockIdx.x; // Batch index
    
    const float* x_sample = gemm_out + b * out_features;
    const float* orig_x_sample = original_x + b * in_features;
    float* res_sample = result + b * in_features;

    // Step 1: Subtract bias and compute Global Avg Pool
    // We need sum of (x_sample[i] - b_sub[i]) for i in 0..out_features-1
    float sum_val = 0.0f;
    
    // Load data into shared memory if out_features is large, or just register/local memory
    // Since out_features=8192, we can iterate.
    for (int i = tid; i < out_features; i += blockDim.x) {
        sum_val += x_sample[i] - b_sub[i];
    }
    
    // Block reduction for sum
    __shared__ float s_sum[256];
    if (tid < 256) s_sum[tid] = 0.0f;
    __syncthreads();
    
    // Simple tree reduction or just atomic add if block size is small enough? 
    // Let's use a standard parallel reduction in shared memory.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }
    
    // Thread 0 has the total sum
    float global_avg = 0.0f;
    if (tid == 0) {
        global_avg = s_sum[0] / out_features;
        
        // Step 2: LogSumExp
        // LSE(x) = log(sum(exp(x)))
        // Here x is a scalar (global_avg), so LSE is just the value itself? 
        // Wait, LogSumExp on a vector reduces it to a scalar.
        // torch.logsumexp(x, dim=1) where x is (B, 1) -> result is (B, 1).
        // If input to logsumexp is already reduced to a scalar per batch item via mean?
        // No: 
        // x = GlobalAvgPool(...) -> shape (B, 1). The value inside is the mean.
        // Then LogSumExp(dim=1) on a tensor of shape (B, 1).
        // logsumexp([v]) = log(exp(v)) = v.
        // So LogSumExp after GlobalAvgPool (which outputs 1 element per batch) is effectively an identity operation 
        // if the dimension being reduced is the only one left?
        // Let's check PyTorch behavior:
        // a = torch.randn(2, 1)
        # torch.logsumexp(a, dim=1) -> returns shape (2, 1). Value is log(sum(exp(a))) = log(exp(a[0])) = a[0].
        # So yes, LogSumExp on a single-element dimension is identity.
        
        float lse_val = global_avg;
        
        // Step 3: GELU
        float gelu_val = gelu(lse_val);
        
        // Step 4: Broadcast add with original_x
        // result[b, :] = original_x[b, :] + gelu_val
        for (int j = tid; j < in_features; j += blockDim.x) {
            res_sample[j] = orig_x_sample[j] + gelu_val;
        }
    }
}

// Wrapper for the tail kernel
torch::Tensor fused_tail_cuda(torch::Tensor gemm_out, torch::Tensor b_sub, torch::Tensor original_x) {
    int batch_size = gemm_out.size(0);
    int out_features = gemm_out.size(1);
    int in_features = original_x.size(1);
    
    auto result = torch::zeros_like(original_x);
    
    const int block_size = 256;
    // We need shared memory for reduction. Max out_features is 8192, but we only sum them up.
    // The reduction is over out_features. We can use a single thread block per batch item if we handle the loop.
    // But wait, my kernel above uses one block per batch item? 
    // blockIdx.x is batch index. So num_blocks = batch_size.
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Shared memory size: 256 floats for reduction
    int shared_mem_size = block_size * sizeof(float);
    
    tail_kernel<<<batch_size, block_size, shared_mem_size, stream>>>(
        gemm_out.data_ptr<float>(), 
        b_sub.data_ptr<float>(), 
        original_x.data_ptr<float>(), 
        result.data_ptr<float>(), 
        batch_size, out_features, in_features
    );
    
    return result;
}

// Wrapper for Gemm using cuBLAS
torch::Tensor gemm_cuda(torch::Tensor x, torch::Tensor weight) {
    // x: (B, in_features)
    // weight: (out_features, in_features)
    // out = x @ weight.T -> (B, out_features)
    
    int m = x.size(0); // Batch size
    int n = weight.size(0); // Out features
    int k = x.size(1); // In features
    
    auto out = torch::empty({m, n}, x.options());
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, at::cuda::getCurrentCUDAStream());
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // cuBLAS is column-major. 
    // We want C = A * B^T ? No, standard matmul: (B, K) * (K, N)^T ?
    // PyTorch Linear: out = x @ W.T. 
    // In cuBLAS: cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc)
    // Or use column major: cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k, &alpha, W.data(), n, x.data(), k, &beta, out.data(), n);
    // Let's stick to RowMajor for simplicity with PyTorch's default layout.
    
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    
    // A is x (m x k), B is weight (n x k). We want (m x k) * (k x n) = m x n.
    // So we multiply x by weight^T.
    // RowMajor: C(m,n) = A(m,k) * B(k,n)^T ? No.
    // C = A * B^T where A is m x k, B is n x k. Then B^T is k x n. Result m x n.
    
    cublasSgemm(handle, 
                CUBLAS_OP_N,   // Transpose A? No.
                CUBLAS_OP_T,   // Transpose B? Yes, weight is (n,k), we want to multiply by (k,n) effectively?
                               // Wait. x is (m,k). Weight is (n,k). 
                               // PyTorch: x @ W.T. W.T is (k,n). 
                               // So we compute x * W^T.
                m, n, k,       // M, N, K
                &alpha, 
                x.data_ptr<float>(), k,   // A, lda (row major)
                weight.data_ptr<float>(), k, // B, ldb (row major). Note: B is stored as (n,k). 
                                             // In row major, B[i,j] is at i*lda + j.
                                             // We want to treat it as B^T in the formula?
                                             // Actually, cublasSgemm with CUBLAS_OP_T on B means we use B^T.
                                             // If B is stored as (n,k), B^T is (k,n).
                &beta, 
                out.data_ptr<float>(), n); // C, ldc

    cublasDestroy(handle);
    
    return out;
}

// Main fused function
torch::Tensor model_forward_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor b_sub) {
    // 1. Gemm
    auto gemm_out = gemm_cuda(x, weight);
    
    // 2. Tail operations (Subtract, Mean, LogSumExp, GELU, Residual Add)
    auto original_x = x.clone().detach();
    auto result = fused_tail_cuda(gemm_out, b_sub, original_x);
    
    return result;
}

"""

# Load the inline CUDA code
model_module = load_inline(
    name="model_forward_cuda",
    cpp_sources="",
    cuda_sources=optimized_source,
    functions=["model_forward_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_ldflags=["-lcublas"]
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        # We store weight and subtract parameter as buffers or parameters to be passed to CUDA
        # Note: In the original model, 'subtract' is a Parameter. 
        # We will register them as buffers so they are part of the module state but we access raw pointers in CUDA.
        self.register_buffer('weight', torch.empty(out_features, in_features))
        self.register_buffer('b_sub', torch.empty(out_features))
        
    def forward(self, x):
        # The weights and bias need to be initialized by the user or via a setup method?
        # In the original code, they are created in __init__. 
        # Here we use buffers. We assume the caller sets them or we mimic the init.
        # However, to make it a drop-in replacement for the class structure:
        
        # We need to ensure weight and b_sub are on the same device as x
        if not hasattr(self, 'weight') or self.weight.device != x.device:
             # This is a bit hacky for inline loading, but we assume the model is initialized properly.
             pass
            
        return model_module.model_forward_cuda(x, self.weight, self.b_sub)

# Helper to initialize the new model similar to the old one
def create_model_new(in_features, out_features):
    model = ModelNew(in_features, out_features)
    # Initialize weights like nn.Linear (Kaiming uniform) and subtract like torch.randn
    with torch.no_grad():
        # nn.Linear default init: weight ~ U(-sqrt(1/in), sqrt(1/in)), bias=0
        limit = 1.0 / math.sqrt(in_features) if in_features > 0 else 0
        torch.nn.init.uniform_(model.weight, -limit, limit)
        model.b_sub.copy_(torch.randn(out_features))
    return model

import math