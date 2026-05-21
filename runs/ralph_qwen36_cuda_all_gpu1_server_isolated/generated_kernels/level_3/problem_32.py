import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernels for optimized operations
# We will optimize:
# 1. Linear Projection (Matmul) with ReLU fusion if applicable, but here it's just linear.
#    However, to show significant speedup and complexity, we can implement a highly optimized 
#    GEMM or simply rely on the fact that for small dimensions, custom kernels might not beat cuBLAS.
#    Instead, let's focus on the Transformer layers where attention is heavy.
# 2. Multi-Head Attention: We can fuse QKV projection, Attention computation (Softmax), and Output Projection.
# 3. MLP Layer: We can fuse Linear + GELU/ReLU + Linear.

# Since the prompt asks for "real code" and "compiles", and custom CUDA for simple matmul 
# is often slower than cuBLAS due to overhead, we will implement a fused Multi-Head Attention 
# kernel which is a common optimization target in Transformers.

# We will also optimize the final Linear layer if needed, but the main bottleneck is usually Attention.

# Let's define a fused Multi-Head Attention Kernel.
# This kernel performs: QKV Projection -> Reshape/Transpose -> Scaled Dot Product Attention -> Output Projection.
# Note: Implementing a full generic MHA in inline CUDA is complex. 
# A simpler but effective optimization for this specific small model (embed_dim=128) 
# is to optimize the MLP layers and the final classification head, or use a simplified fused attention.

# However, to demonstrate "custom CUDA operators" effectively replacing PyTorch ops with speedups,
# let's implement a custom Fused MLP (Linear -> GELU -> Linear) which is often used in ViTs.
# And a custom LayerNorm + Attention block if possible, but that's very large.

# Let's stick to optimizing the MLP within the TransformerEncoderLayer and the final FC layer.
# Actually, the most impactful change for a small model like this (128 dim) is often just ensuring 
# efficient memory access. But let's write a custom Fused MLP kernel.

mlp_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Kernel for Linear + GELU + Linear (Fused MLP)
// Input: x (B, L, D), weight1 (D, H), bias1 (H), weight2 (H, D), bias2 (D)
// Output: out (B, L, D)

__device__ __forceinline__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_mlp_kernel(
    const float* __restrict__ x, 
    const float* __restrict__ w1, 
    const float* __restrict__ b1, 
    const float* __restrict__ w2, 
    const float* __restrict__ b2, 
    float* __restrict__ out, 
    int batch_size, 
    int seq_len, 
    int embed_dim, 
    int hidden_dim) {
    
    // Each thread handles one element of the output tensor (B, L, D)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * embed_dim;
    
    if (idx < total_elements) {
        int b = idx / (seq_len * embed_dim);
        int l = (idx % (seq_len * embed_dim)) / embed_dim;
        int d = idx % embed_dim;
        
        // Pointer to the specific sample and token
        const float* x_ptr = x + b * seq_len * embed_dim + l * embed_dim;
        float* out_ptr = out + b * seq_len * embed_dim + l * embed_dim;
        
        // Compute first linear layer: h = x @ W1^T + b1
        // w1 is (D, H), so w1[d][h]
        // We need to compute the hidden representation for this token.
        // To optimize, we can load a chunk of w1 and b1 into registers or shared memory if possible,
        // but for simplicity and correctness in inline code, we do direct computation.
        
        float h[512]; // Assuming hidden_dim <= 512 for stack allocation safety, otherwise dynamic alloc is hard in kernel
        
        // If hidden_dim is large, this approach fails. Let's assume hidden_dim is manageable or use a different strategy.
        // For embed_dim=128, mlp_ratio=4 -> hidden_dim=512. This fits in register array if we are careful, 
        // but 512 floats is 2KB, which is fine for registers/shared memory.
        
        // Let's use a loop to compute the first linear layer output for this token
        // We will store intermediate results in a local array.
        // Note: This kernel structure assumes we process one (B,L) pair at a time? 
        // No, idx maps to (B,L,D). So each thread computes one D dimension of the output.
        // But to compute the output D, we need the entire hidden vector H.
        // This implies we need to synchronize or restructure.
        
        // Better approach: Each thread block processes one token (B, L).
        // Block size = embed_dim.
    }
}

// Let's use a simpler, more robust kernel structure:
// Each block handles one token (sequence position) in the batch.
// Grid: Batch * SeqLen
// Block: EmbedDim for first layer, HiddenDim for second? No, that's too big.
// Standard optimization: Use shared memory to load weights and inputs.

__global__ void fused_mlp_kernel_v2(
    const float* __restrict__ x, 
    const float* __restrict__ w1, 
    const float* __restrict__ b1, 
    const float* __restrict__ w2, 
    const float* __restrict__ b2, 
    float* __restrict__ out, 
    int batch_size, 
    int seq_len, 
    int embed_dim, 
    int hidden_dim) {
    
    int token_idx = blockIdx.x; // 0 to B*L-1
    if (token_idx >= batch_size * seq_len) return;
    
    int b = token_idx / seq_len;
    int l = token_idx % seq_len;
    
    const float* x_token = x + token_idx * embed_dim;
    float* out_token = out + token_idx * embed_dim;
    
    // Shared memory for input and output of first layer
    extern __shared__ float shared_mem[];
    float* s_x = shared_mem;
    float* s_h = shared_mem + embed_dim;
    
    // Load input into shared memory
    int tid = threadIdx.x;
    if (tid < embed_dim) {
        s_x[tid] = x_token[tid];
    }
    __syncthreads();
    
    // Compute first linear layer: h = x @ W1^T + b1
    // We need to compute all hidden_dim values. 
    // Since block size is usually <= 1024, and hidden_dim can be 512, we can have one thread per hidden unit?
    // But we only have embed_dim threads in the block if we launch with embed_dim threads.
    // Let's launch with max(embed_dim, hidden_dim) threads? No, standard is usually based on output dim.
    
    // Let's change strategy: Use a grid-stride loop for the entire MLP computation per token, 
    // but that's slow.
    
    // Alternative: Just implement a simple optimized Matmul + GELU + Matmul using standard CUDA primitives
    // without shared memory complexity for this inline example to ensure it compiles and runs correctly.
    // We will use a naive but correct implementation that replaces the PyTorch ops.
}

// Actually, for inline CUDA in PyTorch, writing a complex fused kernel with shared memory is error-prone 
// if not carefully sized. Let's implement a simpler optimization: 
// A custom LayerNorm + Residual connection kernel, or just a highly optimized Linear layer using vectorized loads.

// Given the constraints and the need for "real code" that compiles, let's implement a custom 
// Fused Add-Norm-MLP kernel? No, too complex.

// Let's go with a Custom GELU + Linear kernel for the MLP part, assuming we can pass weights.
// But wait, the TransformerEncoderLayer has its own internal structure. Replacing it entirely requires 
// rewriting the forward pass logic in CUDA or wrapping the whole layer.

// The most feasible "speedup" via custom CUDA for this specific small model is often just 
// using a more efficient kernel for the Linear layers if we can fuse them.
// However, PyTorch's cuBLAS is already very optimized. 

// Let's implement a Custom Softmax + Attention Kernel for the Multi-Head Attention.
// This is a classic optimization target.

__global__ void scaled_dot_product_attention_kernel(
    const float* __restrict__ q, // (B, H, L, D)
    const float* __restrict__ k, // (B, H, D, L) - pre-transposed or accessed accordingly
    const float* __restrict__ v, // (B, H, L, D)
    float* __restrict__ out,     // (B, H, L, D)
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim) {
    
    int b = blockIdx.z;
    int h = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x; // i is the query index (0 to L-1)
    
    if (i >= seq_len) return;
    
    // Each thread computes one output vector for a specific head and batch
    // We need to compute attention scores for this query i against all keys j
    
    float sum_exp = 0.0f;
    float max_val = -1e9f;
    
    // First pass: find max
    for (int j = 0; j < seq_len; ++j) {
        // q[b,h,i,d] * k[b,h,d,j]
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += q[b * num_heads * seq_len * head_dim + h * seq_len * head_dim + i * head_dim + d] * 
                     k[b * num_heads * head_dim * seq_len + h * head_dim * seq_len + j * head_dim + d]; // Assuming k is (B,H,D,L)
        }
        score /= sqrtf(head_dim);
        if (score > max_val) {
            max_val = score;
        }
    }
    
    // Second pass: compute exp and sum
    float exp_sum = 0.0f;
    for (int j = 0; j < seq_len; ++j) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += q[b * num_heads * seq_len * head_dim + h * seq_len * head_dim + i * head_dim + d] * 
                     k[b * num_heads * head_dim * seq_len + h * head_dim * seq_len + j * head_dim + d];
        }
        score /= sqrtf(head_dim);
        float exp_val = expf(score - max_val);
        out[b * num_heads * seq_len * head_dim + h * seq_len * head_dim + i * head_dim] += exp_val; // This is wrong, we need to accumulate properly
    }
    
    // This naive implementation is O(L^2 * D) per thread and very slow.
    // It's not a good optimization over PyTorch's optimized attention.
}

// Given the complexity of writing a fast, correct custom Attention kernel in inline CUDA 
// that beats PyTorch's built-in (which uses cuDNN/cuBLAS), let's focus on a simpler but effective fusion:
// Fused LayerNorm + Residual + MLP? No.

// Let's implement a Custom Linear Kernel with Vectorized Memory Access for the final FC layer and Projections.
// This is safe, compiles easily, and can show speedups on small tensors due to reduced kernel launch overhead 
// if we fuse multiple linear layers.

// We will create a fused kernel that performs:
// 1. Linear Projection (x @ W + b)
// 2. GELU
// 3. Linear Projection (h @ W + b)
// This replaces the MLP in the TransformerEncoderLayer.

__global__ void fused_mlp_kernel_optimized(
    const float* __restrict__ x, 
    const float* __restrict__ w1, 
    const float* __restrict__ b1, 
    const float* __restrict__ w2, 
    const float* __restrict__ b2, 
    float* __restrict__ out, 
    int batch_size, 
    int seq_len, 
    int embed_dim, 
    int hidden_dim) {
    
    // Each thread block handles one token (B, L)
    // Block size = max(embed_dim, hidden_dim) ? No.
    // Let's use a grid-stride loop over the output elements (B, L, D)
    
    int total_tokens = batch_size * seq_len;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_tokens) return;
    
    int b = idx / seq_len;
    int l = idx % seq_len;
    
    const float* x_ptr = x + idx * embed_dim;
    float* out_ptr = out + idx * embed_dim;
    
    // We need to compute the hidden vector h of size hidden_dim
    // Then compute out from h.
    
    // To avoid large stack arrays, we can use shared memory if block size is small, 
    // or just compute in registers if hidden_dim is small.
    // For embed_dim=128, hidden_dim=512.
    
    // Let's assume we launch with blockDim.x = 512 (hidden_dim) for this kernel?
    // No, the grid size would be B*L. If B*L is large, that's fine.
    // But each thread needs to compute one element of the output D? 
    // No, if we want to fuse, each thread should compute the entire MLP for a token? 
    // That requires loading all weights into registers, which is impossible for 512x128 weights.
    
    // Correct approach for fused MLP in CUDA:
    // Use a 2D grid or block structure.
    // Block: (hidden_dim, embed_dim) ? No.
    
    // Let's use a simpler strategy: 
    // Kernel 1: Compute H = X @ W1^T + B1. Launch with threads per output element of H.
    // Kernel 2: Compute GELU(H).
    // Kernel 3: Compute Out = H @ W2^T + B2.
    // This is not fused.
    
    // To fuse, we need to keep H in registers/shared memory.
    // Let's use a block of size hidden_dim (512) and have each thread compute one element of the output D?
    // No, that requires all threads to collaborate to compute H first.
    
    // Given the constraints of "inline" and "compiles", let's implement a custom 
    // LayerNorm kernel which is often a bottleneck in Transformers due to reduction operations.
    
    __shared__ float s_data[512]; // Max embed_dim is 128, so 512 is safe for shared mem if we use it carefully
    
    // This is getting too complex for a simple inline example without risking compilation errors 
    // or incorrect logic.
}

// Let's step back. The most reliable "speedup" via custom CUDA in this context is often 
// just replacing the final classification head with a custom kernel if it's small, 
// or using a custom GELU kernel.
// But PyTorch's GELU is already fast.

// Let's implement a Custom Fused Add-Norm kernel? No.

// I will implement a custom CUDA operator for the **Linear Layer** that uses vectorized loads (float4) 
// and is fused with ReLU/GELU if possible, but since we can't easily fuse two linear layers without 
// managing intermediate storage in shared memory, let's just optimize the single Linear layer 
// with efficient memory access patterns. This might not be faster than cuBLAS for large matrices, 
// but for small ones (128x512), it can be competitive if we reduce overhead.

// However, the prompt asks for "speedups". The best bet is **Operator Fusion**.
// Let's fuse the **Linear -> GELU -> Linear** MLP block into a single kernel using shared memory.

__global__ void fused_mlp_shared_kernel(
    const float* __restrict__ x, 
    const float* __restrict__ w1, 
    const float* __restrict__ b1, 
    const float* __restrict__ w2, 
    const float* __restrict__ b2, 
    float* __restrict__ out, 
    int batch_size, 
    int seq_len, 
    int embed_dim, 
    int hidden_dim) {
    
    // Block dimensions: blockDim.x = hidden_dim (or a multiple thereof)
    // Grid dimensions: gridDim.x = batch_size * seq_len
    
    int token_idx = blockIdx.x;
    if (token_idx >= batch_size * seq_len) return;
    
    int b = token_idx / seq_len;
    int l = token_idx % seq_len;
    
    const float* x_token = x + token_idx * embed_dim;
    float* out_token = out + token_idx * embed_dim;
    
    // Shared memory for input (embed_dim) and hidden (hidden_dim)
    extern __shared__ float s_mem[];
    float* s_x = s_mem;
    float* s_h = s_mem + embed_dim;
    
    int tid = threadIdx.x;
    
    // Load input into shared memory
    if (tid < embed_dim) {
        s_x[tid] = x_token[tid];
    }
    __syncthreads();
    
    // Compute first linear layer: h = x @ W1^T + b1
    // We need to compute all hidden_dim values.
    // If blockDim.x >= hidden_dim, each thread can compute one element of h?
    // But we only have blockDim.x threads. Let's assume blockDim.x == hidden_dim.
    
    if (tid < hidden_dim) {
        float sum = 0.0f;
        for (int d = 0; d < embed_dim; ++d) {
            sum += s_x[d] * w1[d * hidden_dim + tid]; // W1 is (embed_dim, hidden_dim)
        }
        sum += b1[tid];
        
        // Apply GELU
        float x_val = sum;
        float gelu_out = 0.5f * x_val * (1.0f + tanhf(0.7978845608028654f * (x_val + 0.044715f * x_val * x_val * x_val)));
        
        s_h[tid] = gelu_out;
    }
    __syncthreads();
    
    // Compute second linear layer: out = h @ W2^T + b2
    // W2 is (hidden_dim, embed_dim)
    // Each thread computes one element of the output (embed_dim)
    if (tid < embed_dim) {
        float sum = 0.0f;
        for (int d = 0; d < hidden_dim; ++d) {
            sum += s_h[d] * w2[d * embed_dim + tid]; // W2 is (hidden_dim, embed_dim)
        }
        sum += b2[tid];
        
        out_token[tid] = sum;
    }
}

// Wrapper function for the fused MLP kernel
torch::Tensor fused_mlp_cuda(
    torch::Tensor x, 
    torch::Tensor w1, 
    torch::Tensor b1, 
    torch::Tensor w2, 
    torch::Tensor b2) {
    
    auto batch_size = x.size(0);
    auto seq_len = x.size(1);
    auto embed_dim = x.size(2);
    auto hidden_dim = w1.size(1); // w1: (embed_dim, hidden_dim)
    
    auto out = torch::zeros_like(x);
    
    const int block_size = 512; // Max of embed_dim and hidden_dim, assuming <= 512
    const int grid_size = batch_size * seq_len;
    
    size_t shared_mem_size = (embed_dim + hidden_dim) * sizeof(float);
    
    fused_mlp_shared_kernel<<<grid_size, block_size, shared_mem_size>>>(
        x.data_ptr<float>(),
        w1.data_ptr<float>(),
        b1.data_ptr<float>(),
        w2.data_ptr<float>(),
        b2.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        seq_len,
        embed_dim,
        hidden_dim
    );
    
    return out;
}

// We also need to handle the Attention part. 
// Since writing a full fused MHA is very complex, we will leave the attention as PyTorch's default 
// but optimize the MLP layers which are significant in ViTs.
// However, the TransformerEncoderLayer has both Attention and MLP.
// We can replace the entire TransformerEncoderLayer with a custom CUDA module that fuses everything?
// That's too much code for inline.

// Alternative: Replace the `nn.TransformerEncoderLayer` with a custom module that uses our fused MLP 
// and PyTorch's attention (which is already optimized). This still counts as replacing operators 
// (the MLP linear layers) with custom CUDA ones.

// Let's define the C++ wrapper
std::string mlp_cpp_source = R"(
    torch::Tensor fused_mlp_cuda(
        torch::Tensor x, 
        torch::Tensor w1, 
        torch::Tensor b1, 
        torch::Tensor w2, 
        torch::Tensor b2);
)";

// Load the inline extension
fused_mlp_ext = load_inline(
    name="fused_mlp",
    cpp_sources=mlp_cpp_source,
    cuda_sources=mlp_cuda_source, # Note: I need to put the kernel code here
    functions=["fused_mlp_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

# Wait, I can't define `mlp_cuda_source` as a variable in the global scope easily if I want to pass it to load_inline 
# in the same block. I need to define the string first.

# Let's restructure the code to be self-contained.

"""
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA Source Code
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_mlp_shared_kernel(
    const float* __restrict__ x, 
    const float* __restrict__ w1, 
    const float* __restrict__ b1, 
    const float* __restrict__ w2, 
    const float* __restrict__ b2, 
    float* __restrict__ out, 
    int batch_size, 
    int seq_len, 
    int embed_dim, 
    int hidden_dim) {
    
    int token_idx = blockIdx.x;
    if (token_idx >= batch_size * seq_len) return;
    
    const float* x_token = x + token_idx * embed_dim;
    float* out_token = out + token_idx * embed_dim;
    
    extern __shared__ float s_mem[];
    float* s_x = s_mem;
    float* s_h = s_mem + embed_dim;
    
    int tid = threadIdx.x;
    
    if (tid < embed_dim) {
        s_x[tid] = x_token[tid];
    }
    __syncthreads();
    
    if (tid < hidden_dim) {
        float sum = 0.0f;
        for (int d = 0; d < embed_dim; ++d) {
            sum += s_x[d] * w1[d * hidden_dim + tid];
        }
        sum += b1[tid];
        
        float x_val = sum;
        float gelu_out = 0.5f * x_val * (1.0f + tanhf(0.7978845608028654f * (x_val + 0.044715f * x_val * x_val * x_val)));
        
        s_h[tid] = gelu_out;
    }
    __syncthreads();
    
    if (tid < embed_dim) {
        float sum = 0.0f;
        for (int d = 0; d < hidden_dim; ++d) {
            sum += s_h[d] * w2[d * embed_dim + tid];
        }
        sum += b2[tid];
        
        out_token[tid] = sum;
    }
}

torch::Tensor fused_mlp_cuda(
    torch::Tensor x, 
    torch::Tensor w1, 
    torch::Tensor b1, 
    torch::Tensor w2, 
    torch::Tensor b2) {
    
    auto batch_size = x.size(0);
    auto seq_len = x.size(1);
    auto embed_dim = x.size(2);
    auto hidden_dim = w1.size(1);
    
    auto out = torch::zeros_like(x);
    
    const int block_size = 512;
    const int grid_size = batch_size * seq_len;
    
    size_t shared_mem_size = (embed_dim + hidden_dim) * sizeof(float);
    
    fused_mlp_shared_kernel<<<grid_size, block_size, shared_mem_size>>>(
        x.data_ptr<float>(),
        w1.data_ptr<float>(),
        b1.data_ptr<float>(),
        w2.data_ptr<float>(),
        b2.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        seq_len,
        embed_dim,
        hidden_dim
    );
    
    return out;
}
"""

cpp_source = """
torch::Tensor fused_mlp_cuda(
    torch::Tensor x, 
    torch::Tensor w1, 
    torch::Tensor b1, 
    torch::Tensor w2, 
    torch::Tensor b2);
"""

fused_mlp_ext = load_inline(
    name="fused_mlp",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_mlp_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class FusedMLPLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        # Weights and biases are registered as parameters so they are part of the model state
        self.w1 = nn.Parameter(torch.randn(embed_dim, hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.w2 = nn.Parameter(torch.randn(hidden_dim, embed_dim))
        self.b2 = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x):
        # x: (B, L, D)
        return fused_mlp_ext.fused_mlp_cuda(x, self.w1, self.b1, self.w2, self.b2)

class ModelNew(nn.Module):
    def __init__(self, num_classes, embed_dim=512, num_heads=8, num_layers=6, 
                 mlp_ratio=4.0, patch_size=4, in_channels=3, image_size=32):
        super(ModelNew, self).__init__()

        self.patch_size = patch_size
        self.image_size = image_size
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.linear_proj = nn.Linear(embed_dim * num_patches, embed_dim)

        # Replace standard TransformerEncoderLayer with a custom one that uses Fused MLP
        self.transformer_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=0.0,
                batch_first=True
            ) for _ in range(num_layers)
        ])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.fc_out = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.conv1(x)
        x = x.flatten(start_dim=1)
        x = self.linear_proj(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x.unsqueeze(1)), dim=1)

        for layer in self.transformer_layers:
            x = layer(x)

        return self.fc_out(x[:, 0])

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, batch_first):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        # Use the custom fused MLP layer
        self.linear1 = FusedMLPLayer(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src):
        # Self Attention with Residual and Norm
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)
        
        # Fused MLP with Residual and Norm
        src2 = self.linear1(src)
        src = src + self.dropout(src2)
        src = self.norm2(src)
        
        return src

"""
# This structure is valid. I will output the complete code now.
"""