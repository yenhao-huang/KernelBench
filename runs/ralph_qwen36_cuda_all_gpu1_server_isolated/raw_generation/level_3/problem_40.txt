```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for GRU cell operations.
# This implementation fuses the input gate (i), update gate (z), and reset gate (r) calculations,
# along with the candidate hidden state computation into a single kernel pass over the data
# to minimize memory bandwidth bottlenecks associated with reading/writing intermediate tensors.

gru_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Helper for matrix multiplication: C = A * B^T + bias
// A: (M, K), B: (N, K) -> C: (M, N)
// We assume row-major storage in PyTorch tensors.
// For GRU, we typically compute:
// i = x @ W_ih^T + h @ W_hh^T + b_ih + b_hh
// where x is (batch, input_size), h is (batch, hidden_size)
// W_ih is (3*hidden_size, input_size), W_hh is (3*hidden_size, hidden_size)

__global__ void gru_cell_forward_kernel(
    const float* x,       // [batch_size, input_size]
    const float* h_prev,  // [batch_size, hidden_size]
    const float* w_ih,    // [3 * hidden_size, input_size]
    const float* w_hh,    // [3 * hidden_size, hidden_size]
    const float* b_ih,    // [3 * hidden_size]
    const float* b_hh,    // [3 * hidden_size]
    float* z_out,         // [batch_size, hidden_size] (output h_t)
    int batch_size,
    int input_size,
    int hidden_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const float* x_row = x + idx * input_size;
    const float* h_row = h_prev + idx * hidden_size;
    
    // Output row
    float* z_row = z_out + idx * hidden_size;

    // We need to compute the three gates: reset (r), update (z), and candidate (n).
    // Let's denote the combined weights as W_ih and W_hh.
    // The output of linear layers is 3*hidden_size.
    // We split this into r, z, n each of size hidden_size.
    
    // To optimize, we can compute the dot products for all three gates in parallel threads or use shared memory.
    // Given the constraints and simplicity, a straightforward approach where each thread block handles one batch item
    // might be too slow if block size is small, but here we map 1 thread per batch element? 
    // No, that would require massive parallelism inside for the dot product.
    // Better: Map threads to hidden_size dimensions within a batch item.
    
    // Let's change strategy: Each thread computes one element of the output vector for a specific batch item.
    // We need to aggregate contributions from input_size and hidden_size.
    // This requires synchronization or atomic adds if we split across threads per batch.
    // For simplicity and correctness in inline code without complex shared memory management, 
    // we will use a grid-stride loop where each thread computes a partial sum for one output element.
    
    // However, standard cuBLAS is highly optimized. Writing a custom matmul kernel from scratch in inline CUDA 
    // that beats cuBLAS for these dimensions (128x256) is extremely difficult and often slower due to lack of tensor cores usage 
    // if not carefully tuned with shared memory tiling.
    
    // Alternative Strategy: Use the fact that PyTorch's GRU is essentially a sequence of Linear + Activation ops.
    // The bottleneck is usually the Linear layers (GEMM).
    // Since we cannot easily replace cuBLAS with a faster custom kernel in a short inline snippet for general GEMM,
    // we will focus on fusing the non-linear activations and bias additions which are memory-bound.
    
    // Actually, let's look at the structure of GRU:
    // r = sigmoid(x @ W_r + h @ U_r + b_r)
    // z = sigmoid(x @ W_z + h @ U_z + b_z)
    // n = tanh(x @ W_n + (r * h) @ U_n + b_n)  <-- Note: reset gate applied to h before matmul with U_n? 
    // Standard GRU formula:
    // r_t = sigma(W_ir x_t + b_ir + W_hr h_{t-1} + b_hr)
    // z_t = sigma(W_iz x_t + b_iz + W_hz h_{t-1} + b_hz)
    // n_t = tanh(W_in x_t + b_in + W_hn (r_t * h_{t-1}) + b_hn)
    // h_t = (1 - z_t) * n_t + z_t * h_{t-1}
    
    // The term (r_t * h_{t-1}) is an element-wise multiplication.
    // This suggests we can fuse:
    // 1. Compute r, z, and the pre-n linear part (x @ W_n + h @ U_n) separately? 
    // No, n depends on r.
    
    // Let's implement a fused kernel that computes the entire GRU cell for one batch item using shared memory for the GEMM parts if possible,
    // or simply rely on the fact that for small hidden sizes (256), we can do it efficiently.
    
    // Given the complexity of writing a high-performance custom GEMM in inline CUDA, 
    // and the instruction to "replace pytorch operators", I will replace the GRU cell logic with a custom kernel 
    // that performs the necessary linear algebra using cuBLAS calls internally if allowed, or simple loops.
    // But wait, load_inline allows including headers. We can use cublas_v2.
    
    // Let's try to fuse the bias additions and activations after the GEMMs.
    // The original PyTorch GRU does:
    // 1. Linear(x) -> 3*hidden
    // 2. Linear(h) -> 3*hidden
    // 3. Add biases
    // 4. Split into r, z, n_pre
    // 5. Apply sigmoid to r, z
    // 6. Element-wise multiply r * h
    // 7. Linear(r*h) -> hidden
    // 8. Add bias
    // 9. Tanh -> n
    // 10. Combine h
    
    // We can fuse steps 3-10 into a single kernel if we assume the GEMMs are done by cuBLAS or standard PyTorch ops, 
    // but the prompt asks to replace operators.
    
    // Let's write a custom GRU cell kernel that uses cuBLAS for the matrix multiplications and then fuses the rest.
    // This is a valid "custom CUDA operator" that replaces the internal logic of the GRU cell.
}

// We will define a Python function that sets up cuBLAS handles and calls kernels.
"""

# Since writing a full high-performance custom GEMM from scratch in inline CUDA is error-prone and likely slower than cuBLAS,
# and fusing activations is the main memory-bound optimization, I will implement a fused GRU cell kernel 
# that uses cuBLAS for the heavy lifting (GEMM) and then performs the element-wise operations in a custom kernel.
# This replaces the standard nn.GRU forward pass logic with a custom one.

gru_cpp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>

// Kernel to fuse bias addition, sigmoid/tanh activations, and final combination for GRU
__global__ void gru_cell_fused_kernel(
    const float* r_in,      // [batch_size, hidden_size] (pre-sigmoid)
    const float* z_in,      // [batch_size, hidden_size] (pre-sigmoid)
    const float* n_in,      // [batch_size, hidden_size] (pre-tanh, includes reset gate effect)
    const float* h_prev,    // [batch_size, hidden_size]
    float* z_out,           // [batch_size, hidden_size] (new hidden state)
    int batch_size,
    int hidden_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * hidden_size) return;

    int h_idx = idx % hidden_size;
    int b_idx = idx / hidden_size;

    float r_val = 1.0f / (1.0f + expf(-r_in[idx])); // sigmoid(r)
    float z_val = 1.0f / (1.0f + expf(-z_in[idx])); // sigmoid(z)
    
    // n_in already contains the result of tanh(W_n * x + W_hn * (r * h_prev))
    // Wait, in standard GRU:
    // n = tanh( W_in x + b_in + W_hn (r_t * h_{t-1}) )
    // My previous split assumed n_in was pre-tanh. Let's assume the caller handles the GEMMs and passes the pre-activation values.
    // However, to be self-contained in the "replacement", I need to know what inputs are passed.
    
    // Let's refine the interface:
    // The custom operator will take x, h_prev, and weights.
    // It will perform GEMMs using cuBLAS internally (or call a helper) and then fuse the rest.
    // But load_inline doesn't easily allow mixing complex C++ logic with CUDA kernels without careful structuring.
    
    // Simpler approach: Replace the entire GRU forward with a custom kernel that does everything.
    // For small hidden sizes, we can implement a tiled GEMM or just use a simple loop if performance isn't critical, 
    // but the prompt asks for speedups.
    
    // Let's stick to the fused activation/bias kernel and assume the GEMMs are done by a helper function in C++.
}

// Helper to perform GRU cell forward using cuBLAS and custom fused kernels
torch::Tensor gru_cell_forward_cuda(
    torch::Tensor x,       // [batch_size, input_size]
    torch::Tensor h_prev,  // [batch_size, hidden_size]
    torch::Tensor w_ih,    // [3 * hidden_size, input_size]
    torch::Tensor w_hh,    // [3 * hidden_size, hidden_size]
    torch::Tensor b_ih,    // [3 * hidden_size]
    torch::Tensor b_hh     // [3 * hidden_size]
) {
    auto batch_size = x.size(0);
    auto input_size = x.size(1);
    auto hidden_size = h_prev.size(1);

    auto device = x.device();
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);

    // Allocate intermediate tensors for the 3 gates (r, z, n_pre)
    // r: [batch, hidden], z: [batch, hidden], n_pre: [batch, hidden]
    auto r = torch::empty({batch_size, hidden_size}, options);
    auto z = torch::empty({batch_size, hidden_size}, options);
    auto n_pre = torch::empty({batch_size, hidden_size}, options);

    cublasHandle_t handle;
    cublasCreate(&handle);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // We need to split w_ih and w_hh into parts for r, z, n.
    // w_ih is [3H, I]. Split into W_ir, W_iz, W_in each [H, I]
    // w_hh is [3H, H]. Split into W_hr, W_hz, W_hn each [H, H]
    
    // To avoid copying weights, we can use pointers with offsets.
    float* w_ih_ptr = w_ih.data_ptr<float>();
    float* w_hh_ptr = w_hh.data_ptr<float>();
    float* b_ih_ptr = b_ih.data_ptr<float>();
    float* b_hh_ptr = b_hh.data_ptr<float>();

    // 1. Compute r: sigmoid(x @ W_ir^T + h @ W_hr^T + b_r)
    // GEMM: r_temp = x @ W_ir^T -> [batch, H]
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                batch_size, hidden_size, input_size, 
                &alpha, 
                x.data_ptr<float>(), input_size, 
                w_ih_ptr, hidden_size, // W_ir is first H rows of w_ih
                0.0f, r.data_ptr<float>(), hidden_size);
    
    // GEMM: r_temp += h @ W_hr^T -> [batch, H]
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                batch_size, hidden_size, hidden_size, 
                &alpha, 
                h_prev.data_ptr<float>(), hidden_size, 
                w_hh_ptr, hidden_size, // W_hr is first H rows of w_hh
                1.0f, r.data_ptr<float>(), hidden_size);
                
    // Add bias b_r (first H elements of b_ih and b_hh)
    // We'll do this in a kernel later or now? Let's fuse activations later.

    // 2. Compute z: sigmoid(x @ W_iz^T + h @ W_hz^T + b_z)
    // GEMM: z_temp = x @ W_iz^T -> [batch, H]
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                batch_size, hidden_size, input_size, 
                &alpha, 
                x.data_ptr<float>(), input_size, 
                w_ih_ptr + hidden_size * input_size, hidden_size, // W_iz is next H rows
                0.0f, z.data_ptr<float>(), hidden_size);
                
    // GEMM: z_temp += h @ W_hz^T -> [batch, H]
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                batch_size, hidden_size, hidden_size, 
                &alpha, 
                h_prev.data_ptr<float>(), hidden_size, 
                w_hh_ptr + hidden_size * hidden_size, hidden_size, // W_hz is next H rows
                1.0f, z.data_ptr<float>(), hidden_size);

    // 3. Compute n: tanh(x @ W_in^T + (r * h) @ W_hn^T + b_n)
    // First compute r * h (element-wise). We need the sigmoid(r) for this.
    // Let's create a temporary tensor for sigmoid(r)
    auto r_sig = torch::empty({batch_size, hidden_size}, options);
    
    // Kernel to compute sigmoid(r) and store in r_sig
    // Also add bias to r and z? No, let's keep it simple.
    
    // We need a kernel to compute:
    // r_sig = sigmoid(r_raw)
    // z_sig = sigmoid(z_raw)
    // n_input = tanh( x@W_in + (r_sig * h) @ W_hn )
    // h_new = (1 - z_sig) * n_input + z_sig * h
    
    // This is getting complex for inline. Let's simplify the custom operator to just replace the 
    // most expensive part: The sequence of GEMMs and activations.
    
    cublasDestroy(handle);
    
    // Return placeholder, we need to implement the full logic in C++/CUDA properly.
    return h_prev; 
}

// Corrected Implementation with proper kernel launch for fused operations
"""

# Let's write a complete, working inline CUDA extension for GRU cell.
# We will define a single Python function that loads the code and exposes the operator.

gru_full_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>

// Kernel to compute sigmoid, tanh, and final GRU update in one pass
// Inputs: r_raw, z_raw, n_raw (pre-activation values for the gates)
// Note: n_raw here is assumed to be the result of W_in*x + W_hn*(sigmoid(r)*h) + bias_n
__global__ void gru_cell_update_kernel(
    const float* r_raw,      // [batch_size, hidden_size]
    const float* z_raw,      // [batch_size, hidden_size]
    const float* n_raw,      // [batch_size, hidden_size] (pre-tanh)
    const float* h_prev,     // [batch_size, hidden_size]
    float* h_new,            // [batch_size, hidden_size]
    int batch_size,
    int hidden_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * hidden_size) return;

    float r_val = 1.0f / (1.0f + expf(-r_raw[idx])); // sigmoid(r)
    float z_val = 1.0f / (1.0f + expf(-z_raw[idx])); // sigmoid(z)
    float n_val = tanhf(n_raw[idx]);                  // tanh(n)
    
    h_new[idx] = (1.0f - z_val) * n_val + z_val * h_prev[idx];
}

// Kernel to add bias to a tensor
__global__ void add_bias_kernel(
    const float* input,
    const float* bias,
    float* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] + bias[idx];
    }
}

torch::Tensor gru_cell_forward_cuda(
    torch::Tensor x,       // [batch_size, input_size]
    torch::Tensor h_prev,  // [batch_size, hidden_size]
    torch::Tensor w_ih,    // [3 * hidden_size, input_size]
    torch::Tensor w_hh,    // [3 * hidden_size, hidden_size]
    torch::Tensor b_ih,    // [3 * hidden_size]
    torch::Tensor b_hh     // [3 * hidden_size]
) {
    auto batch_size = x.size(0);
    auto input_size = x.size(1);
    auto hidden_size = h_prev.size(1);

    auto device = x.device();
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);

    // Intermediate tensors for raw gate values (before activation)
    // r_raw, z_raw, n_raw_pre_tanh
    auto r_raw = torch::empty({batch_size, hidden_size}, options);
    auto z_raw = torch::empty({batch_size, hidden_size}, options);
    auto n_raw = torch::empty({batch_size, hidden_size}, options);

    cublasHandle_t handle;
    cublasCreate(&handle);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;

    float* w_ih_ptr = w_ih.data_ptr<float>();
    float* w_hh_ptr = w_hh.data_ptr<float>();
    float* b_ih_ptr = b_ih.data_ptr<float>();
    float* b_hh_ptr = b_hh.data_ptr<float>();

    // 1. Compute r_raw = x @ W_ir^T + h @ W_hr^T + b_r
    // GEMM: r_temp = x @ W_ir^T
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                batch_size, hidden_size, input_size, 
                &alpha, 
                x.data_ptr<float>(), input_size, 
                w_ih_ptr, hidden_size, 
                0.0f, r_raw.data_ptr<float>(), hidden_size);
    
    // GEMM: r_temp += h @ W_hr^T
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                batch_size, hidden_size, hidden_size, 
                &alpha, 
                h_prev.data_ptr<float>(), hidden_size, 
                w_hh_ptr, hidden_size, 
                1.0f, r_raw.data_ptr<float>(), hidden_size);
                
    // Add bias b_r (first H elements of b_ih and b_hh)
    int block_size = 256;
    int num_blocks = (batch_size * hidden_size + block_size - 1) / block_size;
    
    // We need a kernel that adds two biases? Or we add them sequentially.
    // Let's create a temporary for bias sum if needed, or just add one by one.
    // b_r_total = b_ih[0:H] + b_hh[0:H]
    // To avoid extra allocation, we can add b_ih part first, then b_hh part in the same kernel?
    // Let's write a specific kernel for adding two biases.
    
    // For simplicity, let's assume we add b_ih part now.
    // We'll launch a kernel to add bias to r_raw.
    // But wait, we need to add both b_ih and b_hh parts.
    // Let's create a helper kernel or just do it in the update kernel? No, update is after activation.
    
    // Let's add biases using a simple loop in C++ for small tensors or a kernel.
    // Kernel: add_bias_two_kernel(input, bias1, bias2, output)
    
    auto r_biased = torch::empty({batch_size, hidden_size}, options);
    auto z_biased = torch::empty({batch_size, hidden_size}, options);
    auto n_biased = torch::empty({batch_size, hidden_size}, options);

    // Kernel to add two biases
    auto add_bias_two_source = R"(
__global__ void add_bias_two_kernel(const float* input, const float* bias1, const float* bias2, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] + bias1[idx] + bias2[idx];
    }
}
)";
    
    // Since we can't easily define multiple kernels in one load_inline without careful structuring,
    // and I already defined gru_cell_update_kernel, let's add the bias addition kernel here.
    // Actually, I can just append it to the source string.
    
    // Re-defining source with all kernels
    return h_prev; // Placeholder
}

// We need to restructure the code to include all kernels in one load_inline call.
"""

# Final consolidated source code for the custom GRU operator
custom_gru_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>

__global__ void add_bias_two_kernel(const float* input, const float* bias1, const float* bias2, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] + bias1[idx] + bias2[idx];
    }
}

__global__ void gru_cell_update_kernel(
    const float* r_raw,      // [batch_size, hidden_size]
    const float* z_raw,      // [batch_size, hidden_size]
    const float* n_raw,      // [batch_size, hidden_size] (pre-tanh)
    const float* h_prev,     // [batch_size, hidden_size]
    float* h_new,            // [batch_size, hidden_size]
    int batch_size,
    int hidden_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * hidden_size) return;

    float r_val = 1.0f / (1.0f + expf(-r_raw[idx])); // sigmoid(r)
    float z_val = 1.0f / (1.0f + expf(-z_raw[idx])); // sigmoid(z)
    float n_val = tanhf(n_raw[idx]);                  // tanh(n)
    
    h_new[idx] = (1.0f - z_val) * n_val + z_val * h_prev[idx];
}

torch::Tensor gru_cell_forward_cuda(
    torch::Tensor x,       // [batch_size, input_size]
    torch::Tensor h_prev,  // [batch_size, hidden_size]
    torch::Tensor w_ih,    // [3 * hidden_size, input_size]
    torch::Tensor w_hh,    // [3 * hidden_size, hidden_size]
    torch::Tensor b_ih,    // [3 * hidden_size]
    torch::Tensor b_hh     // [3 * hidden_size]
) {
    auto batch_size = x.size(0);
    auto input_size = x.size(1);
    auto hidden_size = h_prev.size(1);

    auto device = x.device();
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);

    // Intermediate tensors for raw gate values (before activation)
    auto r_raw = torch::empty({batch_size, hidden_size}, options);
    auto z_raw = torch::empty({batch_size, hidden_size}, options);
    auto n_raw = torch::empty({batch_size, hidden_size}, options);

    cublasHandle_t handle;
    cublasCreate(&handle);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;

    float* w_ih_ptr = w_ih.data_ptr<float>();
    float* w_hh_ptr = w_hh.data_ptr<float>();
    float* b_ih_ptr = b_ih.data_ptr<float>();
    float* b_hh_ptr = b_hh.data_ptr<float>();

    // 1. Compute r_raw = x @ W_ir^T + h @ W_hr^T
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                batch_size, hidden_size, input_size, 
                &alpha, 
                x.data_ptr<float>(), input_size, 
                w_ih_ptr, hidden_size, 
                0.0f, r_raw.data_ptr<float>(), hidden_size);
    
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                batch_size, hidden_size, hidden_size, 
                &alpha, 
                h_prev.data_ptr<float>(), hidden_size, 
                w_hh_ptr, hidden_size, 
                1.0f, r_raw.data_ptr<float>(), hidden_size);
                
    // 2. Compute z_raw = x @ W_iz^T + h @ W_hz^T
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                batch_size, hidden_size, input_size, 
                &alpha, 
                x.data_ptr<float>(), input_size, 
                w_ih_ptr + hidden_size * input_size, hidden_size, 
                0.0f, z_raw.data_ptr<float>(), hidden_size);
                
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                batch_size, hidden_size, hidden_size, 
                &alpha, 
                h_prev.data_ptr<float>(), hidden_size, 
                w_hh_ptr + hidden_size * hidden_size, hidden_size, 
                1.0f, z_raw.data_ptr<float>(), hidden_size);

    // 3. Compute n_raw = x @ W_in^T + (sigmoid(r) * h) @ W_hn^T
    // First compute sigmoid(r)
    auto r_sig = torch::empty({batch_size, hidden_size}, options);
    
    // Kernel to compute sigmoid(r)
    auto sig_kernel_source = R"(
__global__ void sigmoid_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}
)";
    
    // We can't define new kernels dynamically inside the function easily without load_inline re-compilation.
    // So we must include all kernels in the initial source string.
    // I will add sigmoid_kernel to the main source string below.
    
    // For now, let's assume we have a sigmoid kernel available.
    // Since I am writing the full code block, I will ensure all kernels are in the source string.
    
    // Compute r_sig = sigmoid(r_raw)
    int total_elements = batch_size * hidden_size;
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    // Launch sigmoid kernel for r
    // We need to define this kernel in the source string.
    
    // Compute n_raw_pre = x @ W_in^T
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                batch_size, hidden_size, input_size, 
                &alpha, 
                x.data_ptr<float>(), input_size, 
                w_ih_ptr + 2 * hidden_size * input_size, hidden_size, 
                0.0f, n_raw.data_ptr<float>(), hidden_size);
                
    // Compute (r_sig * h) @ W_hn^T
    // We need r_sig first.
    
    cublasDestroy(handle);
    
    return h_prev; // Placeholder to force compilation check, logic below is incomplete without sigmoid kernel in source
}

// To make this work, I must include the sigmoid kernel in the main source string passed to load_inline.
"""

# Corrected full source with all kernels
full_gru_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>

__global__ void add_bias_two_kernel(const float* input, const float* bias1, const float* bias2, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] + bias1[idx] + bias2[idx];
    }
}

__global__ void sigmoid_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

__global__ void gru_cell_update_kernel(
    const float* r_raw,      // [batch_size, hidden_size]
    const float* z_raw,      // [batch_size, hidden_size]
    const float* n_raw,      // [batch_size, hidden_size] (pre-tanh)
    const float* h_prev,     // [batch_size, hidden_size]
    float* h_new,            // [batch_size, hidden_size]
    int batch_size,
    int hidden_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * hidden_size) return;

    float r_val = 1.0f / (1.0f + expf(-r_raw[idx])); // sigmoid(r)
    float z_val = 1.0f / (1.0f + expf(-z_raw[idx])); // sigmoid(z)
    float n_val = tanhf(n_raw[idx]);                  // tanh(n)
    
    h_new[idx] = (1.0f - z_val) * n_val + z_val * h_prev[idx];
}

torch::Tensor gru_cell_forward_cuda(
    torch::Tensor x,       // [batch_size, input_size]
    torch::Tensor h_prev,  // [batch_size, hidden_size]
    torch::Tensor w_ih,    // [3 * hidden_size, input_size]
    torch::Tensor w_hh,    // [3 * hidden_size, hidden_size]
    torch::Tensor b_ih,    // [3 * hidden_size]
    torch::Tensor b_hh     // [3 * hidden_size]
) {
    auto batch_size = x.size(0);
    auto input_size = x.size(1);
    auto hidden_size = h_prev.size(1);

    auto device = x.device();
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);

    // Intermediate tensors for raw gate values (before activation)
    auto r_raw = torch::empty({batch_size, hidden_size}, options);
    auto z_raw = torch::empty({batch_size, hidden_size}, options);
    auto n_raw = torch::empty({batch_size, hidden_size}, options);

    cublasHandle_t handle;
    cublasCreate(&handle);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;

    float* w_ih_ptr = w_ih.data_ptr<float>();
    float* w_hh_ptr = w_hh.data_ptr<float>();
    float* b_ih_ptr = b_ih.data_ptr<float>();
    float* b_hh_ptr = b_hh.data_ptr<float>();

    // 1. Compute r_raw = x @ W_ir^T + h @ W_hr^T
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                batch_size, hidden_size, input_size, 
                &alpha, 
                x.data_ptr<float>(), input_size, 
                w_ih_ptr, hidden_size, 
                0.0f, r_raw.data_ptr<float>(), hidden_size);
    
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                batch_size, hidden_size, hidden_size, 
                &alpha, 
                h_prev.data_ptr<float>(), hidden_size, 
                w_hh_ptr, hidden_size, 
                1.0f, r_raw.data_ptr<float>(), hidden_size);
                
    // 2. Compute z_raw = x @ W_iz^T + h @ W_hz^T
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                batch_size, hidden_size, input_size, 
                &alpha, 
                x.data_ptr<float>(), input_size, 
                w_ih_ptr + hidden_size * input_size, hidden_size, 
                0.0f, z_raw.data_ptr<float>(), hidden_size);
                
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                batch_size, hidden_size, hidden_size, 
                &alpha, 
                h_prev.data_ptr<float>(), hidden_size, 
                w_hh_ptr + hidden_size * hidden_size, hidden_size, 
                1.0f, z_raw.data_ptr<float>(), hidden_size);

    // 3. Compute n_raw = x @ W_in^T + (sigmoid(r) * h) @ W_hn^T
    // First compute sigmoid(r)
    auto r_sig = torch::empty({batch_size, hidden_size}, options);
    
    int total_elements = batch_size * hidden_size;
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    sigmoid_kernel<<<num_blocks, block_size>>>(r_raw.data_ptr<float>(), r_sig.data_ptr<float>(), total_elements);
    cudaDeviceSynchronize();

    // Compute n_raw_pre = x @ W_in^T
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                batch_size, hidden_size, input_size, 
                &alpha, 
                x.data_ptr<float>(), input_size, 
                w_ih_ptr + 2 * hidden_size * input_size, hidden_size, 
                0.0f, n_raw.data