```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA implementation for GRU cell operations fused together.
# This replaces the standard nn.GRU with a custom kernel that performs:
# 1. Linear projections for r, z, n gates (W_ih @ x + b_ih) and (W_hh @ h + b_hh)
# 2. Element-wise sigmoid/tanh activations
# 3. Update gate logic: r = sigmoid(...), z = sigmoid(...), n = tanh(...)
# 4. Output calculation: h_new = (1-z)*n + z*h

gru_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Helper for sigmoid
__device__ inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Helper for tanh
__device__ inline float tanh_val(float x) {
    if (x > 20.0f) return 1.0f;
    if (x < -20.0f) return -1.0f;
    return tanhf(x);
}

// Kernel for a single GRU step across all hidden units and directions
// Assumes input x is [batch, input_size]
// Assumes h_prev is [batch, hidden_size]
// Weights are flattened: W_ih is [3*hidden_size, input_size], W_hh is [3*hidden_size, hidden_size]
// Biases are [3*hidden_size]
__global__ void gru_step_kernel(
    const float* x,           // [batch, input_size]
    const float* h_prev,      // [batch, hidden_size]
    const float* w_ih,        // [3*hidden_size, input_size]
    const float* b_ih,        // [3*hidden_size]
    const float* w_hh,        // [3*hidden_size, hidden_size]
    const float* b_hh,        // [3*hidden_size]
    float* h_new,             // [batch, hidden_size]
    int batch_size,
    int input_size,
    int hidden_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const float* x_row = x + idx * input_size;
    const float* h_prev_row = h_prev + idx * hidden_size;
    float* h_new_row = h_new + idx * hidden_size;

    // We compute the 3 gates: reset (r), update (z), candidate (n)
    // Each gate has hidden_size outputs.
    // Total intermediate values needed: 3 * hidden_size
    
    // Shared memory for weights and biases to reduce global memory access? 
    // Given hidden_size is small (256), we can just compute directly or use registers.
    // Let's stick to direct computation for simplicity and correctness, optimizing memory coalescing.

    // Precompute W_ih @ x + b_ih for all 3 gates
    // This is a matrix-vector multiplication: (3*H) x I -> (3*H)
    // We can split this into 3 parts or compute all at once.
    
    float r_vals[256]; // Max hidden size assumed 256 for stack allocation, otherwise dynamic alloc needed. 
                       // For general case, we might need to be careful with stack size. 
                       // Let's assume hidden_size <= 1024 and use dynamic shared memory or just registers if small.
                       // To be safe and generic, let's avoid large stack arrays if hidden_size is large.
                       // However, for this specific problem, we can iterate.

    // Actually, doing the matmul inside the thread loop over hidden units is better for coalescing 
    // if we structure it right, but here each thread handles one batch item.
    // Inside one batch item, we need to compute 3*hidden_size outputs.
    
    // Let's use a temporary buffer on stack if hidden_size is small enough, or just compute on fly.
    // Given the constraints and typical sizes, let's assume hidden_size fits in registers/shared.
    // To be robust, we will compute gate by gate or use shared memory for the weight matrix row access?
    // No, W_ih is [3H, I]. Accessing W_ih[k * input_size + j] is strided if k changes fast.
    
    // Better approach: Each thread computes one output element of the 3*H vector? 
    // No, that would require H threads per batch item, which is too many.
    // Current setup: 1 thread per batch item. It must compute all 3*H values.
    
    // Optimization: Use shared memory to load W_ih and W_hh rows? 
    // Since I (input_size) can be large, loading the whole row into shared mem is expensive.
    // However, we access W_ih sequentially for each gate if we loop over hidden units.
    
    // Let's compute r, z, n vectors.
    
    float r_sum[256]; 
    float z_sum[256];
    float n_sum[256];

    // Initialize sums
    #pragma unroll
    for (int k = 0; k < hidden_size; ++k) {
        r_sum[k] = b_ih[k];
        z_sum[k] = b_ih[hidden_size + k];
        n_sum[k] = b_ih[2 * hidden_size + k];
    }

    // Compute W_ih @ x
    // We need to add w_ih[k][j] * x[j] to r_sum[k], etc.
    // To optimize, we can loop over input features j and update all hidden units k.
    // This allows coalesced access to x (if x is contiguous) and strided access to W.
    
    for (int j = 0; j < input_size; ++j) {
        float x_val = x_row[j];
        #pragma unroll
        for (int k = 0; k < hidden_size; ++k) {
            r_sum[k] += w_ih[k * input_size + j] * x_val;
            z_sum[k] += w_ih[(hidden_size + k) * input_size + j] * x_val;
            n_sum[k] += w_ih[(2 * hidden_size + k) * input_size + j] * x_val;
        }
    }

    // Compute W_hh @ h_prev
    // Add to the sums
    for (int j = 0; j < hidden_size; ++j) {
        float h_val = h_prev_row[j];
        #pragma unroll
        for (int k = 0; k < hidden_size; ++k) {
            r_sum[k] += w_hh[k * hidden_size + j] * h_val;
            z_sum[k] += w_hh[(hidden_size + k) * hidden_size + j] * h_val;
            n_sum[k] += w_hh[(2 * hidden_size + k) * hidden_size + j] * h_val;
        }
    }

    // Apply activations and compute new hidden state
    // r = sigmoid(r_sum)
    // z = sigmoid(z_sum)
    // n = tanh(n_sum)
    // h_new = (1 - z) * n + z * h_prev
    
    #pragma unroll
    for (int k = 0; k < hidden_size; ++k) {
        float r = sigmoid(r_sum[k]);
        float z = sigmoid(z_sum[k]);
        float n = tanh_val(n_sum[k]);
        
        h_new_row[k] = (1.0f - z) * n + z * h_prev_row[k];
    }
}

// Kernel to process all layers and directions
// Input x: [seq_len, batch, input_size] or [batch, seq_len, input_size] depending on batch_first
// We assume the caller handles the iteration over time steps and layers.
// This kernel processes ONE time step for ONE layer for ALL directions combined? 
// No, GRU is usually implemented as a loop over time.
// The standard nn.GRU loops: for t in seq_len: for l in num_layers: h_l_t = gru_cell(h_l_{t-1}, x_t)

// We will provide a kernel that processes one layer's update for all directions at once?
// Actually, it's easier to have the Python side loop over time and layers, calling a kernel 
// that handles the batch and hidden_size for a specific direction.
// But wait, bidirectional means we have forward and backward passes.
// The standard GRU implementation in PyTorch handles this internally.
// To replace it completely, we need to mimic the structure.

// Let's create a kernel that runs one full GRU layer step (one time step) for all directions.
// Inputs:
// x_t: [batch, input_size] (for current time step)
// h_prev_fwd: [batch, hidden_size]
// h_prev_bwd: [batch, hidden_size]
// Weights for forward and backward
// Outputs:
// h_new_fwd: [batch, hidden_size]
// h_new_bwd: [batch, hidden_size]

__global__ void gru_layer_step_kernel(
    const float* x_t,           // [batch, input_size]
    const float* h_prev_fwd,    // [batch, hidden_size]
    const float* h_prev_bwd,    // [batch, hidden_size]
    
    // Forward weights
    const float* w_ih_fwd,      // [3*hidden_size, input_size]
    const float* b_ih_fwd,      // [3*hidden_size]
    const float* w_hh_fwd,      // [3*hidden_size, hidden_size]
    const float* b_hh_fwd,      // [3*hidden_size]
    
    // Backward weights
    const float* w_ih_bwd,      // [3*hidden_size, input_size]
    const float* b_ih_bwd,      // [3*hidden_size]
    const float* w_hh_bwd,      // [3*hidden_size, hidden_size]
    const float* b_hh_bwd,      // [3*hidden_size]
    
    float* h_new_fwd,           // [batch, hidden_size]
    float* h_new_bwd,           // [batch, hidden_size]
    
    int batch_size,
    int input_size,
    int hidden_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    // Process Forward Direction
    gru_step_kernel<<<1, 1>>>(
        x_t, h_prev_fwd, w_ih_fwd, b_ih_fwd, w_hh_fwd, b_hh_fwd, 
        h_new_fwd + idx * hidden_size, 1, input_size, hidden_size
    ); // Note: launching a kernel from within a kernel is not standard CUDA (unless dynamic parallelism enabled). 
       // Instead, we inline the logic or use a helper function.

    // Since we can't easily launch kernels from kernels without complex setup, let's rewrite the logic inline for both directions in one thread per batch item?
    // That would mean 1 thread handles 2 hidden states (fwd and bwd).
    // This is efficient.
}

// Revised Kernel: One thread per batch item, computes both forward and backward GRU steps
__global__ void gru_bidirectional_step_kernel(
    const float* x_t,           // [batch, input_size]
    const float* h_prev_fwd,    // [batch, hidden_size]
    const float* h_prev_bwd,    // [batch, hidden_size]
    
    // Forward weights
    const float* w_ih_fwd,      // [3*hidden_size, input_size]
    const float* b_ih_fwd,      // [3*hidden_size]
    const float* w_hh_fwd,      // [3*hidden_size, hidden_size]
    const float* b_hh_fwd,      // [3*hidden_size]
    
    // Backward weights
    const float* w_ih_bwd,      // [3*hidden_size, input_size]
    const float* b_ih_bwd,      // [3*hidden_size]
    const float* w_hh_bwd,      // [3*hidden_size, hidden_size]
    const float* b_hh_bwd,      // [3*hidden_size]
    
    float* h_new_fwd,           // [batch, hidden_size]
    float* h_new_bwd,           // [batch, hidden_size]
    
    int batch_size,
    int input_size,
    int hidden_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const float* x_row = x_t + idx * input_size;
    const float* h_fwd_row = h_prev_fwd + idx * hidden_size;
    const float* h_bwd_row = h_prev_bwd + idx * hidden_size;
    float* h_new_fwd_row = h_new_fwd + idx * hidden_size;
    float* h_new_bwd_row = h_new_bwd + idx * hidden_size;

    // Compute Forward GRU Step
    float r_sum[256]; 
    float z_sum[256];
    float n_sum[256];

    #pragma unroll
    for (int k = 0; k < hidden_size; ++k) {
        r_sum[k] = b_ih_fwd[k];
        z_sum[k] = b_ih_fwd[hidden_size + k];
        n_sum[k] = b_ih_fwd[2 * hidden_size + k];
    }

    for (int j = 0; j < input_size; ++j) {
        float x_val = x_row[j];
        #pragma unroll
        for (int k = 0; k < hidden_size; ++k) {
            r_sum[k] += w_ih_fwd[k * input_size + j] * x_val;
            z_sum[k] += w_ih_fwd[(hidden_size + k) * input_size + j] * x_val;
            n_sum[k] += w_ih_fwd[(2 * hidden_size + k) * input_size + j] * x_val;
        }
    }

    for (int j = 0; j < hidden_size; ++j) {
        float h_val = h_fwd_row[j];
        #pragma unroll
        for (int k = 0; k < hidden_size; ++k) {
            r_sum[k] += w_hh_fwd[k * hidden_size + j] * h_val;
            z_sum[k] += w_hh_fwd[(hidden_size + k) * hidden_size + j] * h_val;
            n_sum[k] += w_hh_fwd[(2 * hidden_size + k) * hidden_size + j] * h_val;
        }
    }

    #pragma unroll
    for (int k = 0; k < hidden_size; ++k) {
        float r = sigmoid(r_sum[k]);
        float z = sigmoid(z_sum[k]);
        float n = tanh_val(n_sum[k]);
        h_new_fwd_row[k] = (1.0f - z) * n + z * h_fwd_row[k];
    }

    // Compute Backward GRU Step
    // Reset sums
    #pragma unroll
    for (int k = 0; k < hidden_size; ++k) {
        r_sum[k] = b_ih_bwd[k];
        z_sum[k] = b_ih_bwd[hidden_size + k];
        n_sum[k] = b_ih_bwd[2 * hidden_size + k];
    }

    for (int j = 0; j < input_size; ++j) {
        float x_val = x_row[j];
        #pragma unroll
        for (int k = 0; k < hidden_size; ++k) {
            r_sum[k] += w_ih_bwd[k * input_size + j] * x_val;
            z_sum[k] += w_ih_bwd[(hidden_size + k) * input_size + j] * x_val;
            n_sum[k] += w_ih_bwd[(2 * hidden_size + k) * input_size + j] * x_val;
        }
    }

    for (int j = 0; j < hidden_size; ++j) {
        float h_val = h_bwd_row[j];
        #pragma unroll
        for (int k = 0; k < hidden_size; ++k) {
            r_sum[k] += w_hh_bwd[k * hidden_size + j] * h_val;
            z_sum[k] += w_hh_bwd[(hidden_size + k) * hidden_size + j] * h_val;
            n_sum[k] += w_hh_bwd[(2 * hidden_size + k) * hidden_size + j] * h_val;
        }
    }

    #pragma unroll
    for (int k = 0; k < hidden_size; ++k) {
        float r = sigmoid(r_sum[k]);
        float z = sigmoid(z_sum[k]);
        float n = tanh_val(n_sum[k]);
        h_new_bwd_row[k] = (1.0f - z) * n + z * h_bwd_row[k];
    }
}

// Main function to run the GRU
torch::Tensor gru_forward_cuda(
    torch::Tensor x,          // [seq_len, batch, input_size] or [batch, seq_len, input_size]
    torch::Tensor h0,         // [num_layers * 2, batch, hidden_size]
    torch::Tensor w_ih_list,  // List of tensors for each layer and direction? Or flattened?
    torch::Tensor b_ih_list,
    torch::Tensor w_hh_list,
    torch::Tensor b_hh_list,
    bool batch_first,
    int num_layers,
    int hidden_size,
    int input_size
) {
    auto device = x.device();
    auto dtype = x.dtype();
    
    // Determine shapes based on batch_first
    int seq_len, batch;
    if (batch_first) {
        batch = x.size(0);
        seq_len = x.size(1);
        // x is [batch, seq_len, input_size] -> need to permute to [seq_len, batch, input_size] for easier processing
        x = x.permute({1, 0, 2});
    } else {
        seq_len = x.size(0);
        batch = x.size(1);
    }

    // h0 shape: [num_layers * 2, batch, hidden_size]
    // Directions: 0 is forward, 1 is backward for each layer
    
    auto output_options = torch::TensorOptions().dtype(dtype).device(device).requires_grad(false);
    torch::Tensor output = torch::empty({seq_len, batch, hidden_size * 2}, output_options); // Concatenated fwd and bwd outputs
    
    // Current hidden states: [num_layers * 2, batch, hidden_size]
    torch::Tensor h_current = h0.clone();

    const int block_size = 256;
    const int num_blocks = (batch + block_size - 1) / block_size;

    for (int t = 0; t < seq_len; ++t) {
        // x_t: [batch, input_size]
        torch::Tensor x_t = x[t]; 
        
        torch::Tensor h_next = torch::empty_like(h_current);
        
        // We need to process each layer. 
        // For layer l, we have forward (idx 2*l) and backward (idx 2*l+1)
        // The input to layer l is the output of layer l-1.
        // For t=0, input is h0. For t>0, input is h_next from previous time step? 
        // No, h_current holds the hidden states for the CURRENT time step before update.
        // We need to compute new hidden states for all layers simultaneously? 
        // No, layer l depends on layer l-1's output at time t.
        
        // So we must loop over layers inside the time step loop.
        
        torch::Tensor h_input = h_current; // Start with h0 for first layer
        
        for (int l = 0; l < num_layers; ++l) {
            int dir_fwd_idx = 2 * l;
            int dir_bwd_idx = 2 * l + 1;
            
            torch::Tensor h_fwd_prev = h_input[dir_fwd_idx]; // [batch, hidden_size]
            torch::Tensor h_bwd_prev = h_input[dir_bwd_idx]; // [batch, hidden_size]
            
            // Weights for this layer and direction
            // w_ih_list[l] is [3*hidden_size, input_size] for forward? 
            // Usually PyTorch stores them in a list. Let's assume the caller passes flattened weights or we extract them.
            // For simplicity in this inline example, let's assume the weights are passed as separate tensors for fwd/bwd per layer.
            // But the function signature above is getting complex. 
            // Let's simplify: The Python wrapper will handle extracting weights and calling the kernel.
            
            // To make this self-contained in one call, we need to pass all weights.
            // Let's assume w_ih_fwd[l], w_ih_bwd[l] etc are passed as tensors in a list or concatenated.
            // Given the complexity of passing lists of tensors to CUDA kernels, it's often easier to 
            // have the Python side loop over layers and call a kernel that handles one layer step.
            
            // Let's change strategy: The Python code will loop over time and layers, calling a kernel for each (t, l).
            // This is less fused but much simpler to implement correctly with PyTorch's tensor management.
            // However, the prompt asks for optimization. Fusing time steps is hard due to dependencies. 
            // Fusing directions is good. Fusing layers is impossible due to dependency.
            
            // So, we will create a kernel `gru_layer_step` that takes x_t, h_fwd_prev, h_bwd_prev and returns h_fwd_next, h_bwd_next.
            // The Python code loops t and l.
        }
    }
    
    // Since the above logic is getting tangled with weight passing, let's define a simpler kernel 
    // that processes one layer step for both directions, and call it from Python.
    
    return output;
}

// Kernel for one layer step (both directions)
__global__ void gru_layer_step_kernel(
    const float* x_t,           // [batch, input_size]
    const float* h_fwd_prev,    // [batch, hidden_size]
    const float* h_bwd_prev,    // [batch, hidden_size]
    
    const float* w_ih_fwd,      // [3*hidden_size, input_size]
    const float* b_ih_fwd,      // [3*hidden_size]
    const float* w_hh_fwd,      // [3*hidden_size, hidden_size]
    const float* b_hh_fwd,      // [3*hidden_size]
    
    const float* w_ih_bwd,      // [3*hidden_size, input_size]
    const float* b_ih_bwd,      // [3*hidden_size]
    const float* w_hh_bwd,      // [3*hidden_size, hidden_size]
    const float* b_hh_bwd,      // [3*hidden_size]
    
    float* h_fwd_next,          // [batch, hidden_size]
    float* h_bwd_next,          // [batch, hidden_size]
    
    int batch_size,
    int input_size,
    int hidden_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const float* x_row = x_t + idx * input_size;
    const float* h_fwd_row = h_fwd_prev + idx * hidden_size;
    const float* h_bwd_row = h_bwd_prev + idx * hidden_size;
    float* h_fwd_next_row = h_fwd_next + idx * hidden_size;
    float* h_bwd_next_row = h_bwd_next + idx * hidden_size;

    // Forward GRU
    float r_sum[256]; 
    float z_sum[256];
    float n_sum[256];

    #pragma unroll
    for (int k = 0; k < hidden_size; ++k) {
        r_sum[k] = b_ih_fwd[k];
        z_sum[k] = b_ih_fwd[hidden_size + k];
        n_sum[k] = b_ih_fwd[2 * hidden_size + k];
    }

    for (int j = 0; j < input_size; ++j) {
        float x_val = x_row[j];
        #pragma unroll
        for (int k = 0; k < hidden_size; ++k) {
            r_sum[k] += w_ih_fwd[k * input_size + j] * x_val;
            z_sum[k] += w_ih_fwd[(hidden_size + k) * input_size + j] * x_val;
            n_sum[k] += w_ih_fwd[(2 * hidden_size + k) * input_size + j] * x_val;
        }
    }

    for (int j = 0; j < hidden_size; ++j) {
        float h_val = h_fwd_row[j];
        #pragma unroll
        for (int k = 0; k < hidden_size; ++k) {
            r_sum[k] += w_hh_fwd[k * hidden_size + j] * h_val;
            z_sum[k] += w_hh_fwd[(hidden_size + k) * hidden_size + j] * h_val;
            n_sum[k] += w_hh_fwd[(2 * hidden_size + k) * hidden_size + j] * h_val;
        }
    }

    #pragma unroll
    for (int k = 0; k < hidden_size; ++k) {
        float r = sigmoid(r_sum[k]);
        float z = sigmoid(z_sum[k]);
        float n = tanh_val(n_sum[k]);
        h_fwd_next_row[k] = (1.0f - z) * n + z * h_fwd_row[k];
    }

    // Backward GRU
    #pragma unroll
    for (int k = 0; k < hidden_size; ++k) {
        r_sum[k] = b_ih_bwd[k];
        z_sum[k] = b_ih_bwd[hidden_size + k];
        n_sum[k] = b_ih_bwd[2 * hidden_size + k];
    }

    for (int j = 0; j < input_size; ++j) {
        float x_val = x_row[j];
        #pragma unroll
        for (int k = 0; k < hidden_size; ++k) {
            r_sum[k] += w_ih_bwd[k * input_size + j] * x_val;
            z_sum[k] += w_ih_bwd[(hidden_size + k) * input_size + j] * x_val;
            n_sum[k] += w_ih_bwd[(2 * hidden_size + k) * input_size + j] * x_val;
        }
    }

    for (int j = 0; j < hidden_size; ++j) {
        float h_val = h_bwd_row[j];
        #pragma unroll
        for (int k = 0; k < hidden_size; ++k) {
            r_sum[k] += w_hh_bwd[k * hidden_size + j] * h_val;
            z_sum[k] += w_hh_bwd[(hidden_size + k) * hidden_size + j] * h_val;
            n_sum[k] += w_hh_bwd[(2 * hidden_size + k) * hidden_size + j] * h_val;
        }
    }

    #pragma unroll
    for (int k = 0; k < hidden_size; ++k) {
        float r = sigmoid(r_sum[k]);
        float z = sigmoid(z_sum[k]);
        float n = tanh_val(n_sum[k]);
        h_bwd_next_row[k] = (1.0f - z) * n + z * h_bwd_row[k];
    }
}

// Python wrapper function
torch::Tensor gru_layer_step_cuda(
    torch::Tensor x_t,
    torch::Tensor h_fwd_prev,
    torch::Tensor h_bwd_prev,
    torch::Tensor w_ih_fwd,
    torch::Tensor b_ih_fwd,
    torch::Tensor w_hh_fwd,
    torch::Tensor b_hh_fwd,
    torch::Tensor w_ih_bwd,
    torch::Tensor b_ih_bwd,
    torch::Tensor w_hh_bwd,
    torch::Tensor b_hh_bwd
) {
    auto batch_size = x_t.size(0);
    auto input_size = x_t.size(1);
    auto hidden_size = h_fwd_prev.size(1);
    
    auto device = x_t.device();
    auto dtype = x_t.dtype();
    
    torch::Tensor h_fwd_next = torch::empty({batch_size, hidden_size}, torch::TensorOptions().dtype(dtype).device(device));
    torch::Tensor h_bwd_next = torch::empty({batch_size, hidden_size}, torch::TensorOptions().dtype(dtype).device(device));
    
    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;
    
    gru_layer_step_kernel<<<num_blocks, block_size>>>(
        x_t.data_ptr<float>(),
        h_fwd_prev.data_ptr<float>(),
        h_bwd_prev.data_ptr<float>(),
        w_ih_fwd.data_ptr<float>(),
        b_ih_fwd.data_ptr<float>(),
        w_hh_fwd.data_ptr<float>(),
        b_hh_fwd.data_ptr<float>(),
        w_ih_bwd.data_ptr<float>(),
        b_ih_bwd.data_ptr<float>(),
        w_hh_bwd.data_ptr<float>(),
        b_hh_bwd.data_ptr<float>(),
        h_fwd_next.data_ptr<float>(),
        h_bwd_next.data_ptr<float>(),
        batch_size, input_size, hidden_size
    );
    
    return {h_fwd_next, h_bwd_next};
}

"""

gru_cpp_source = (
    "torch::Tensor gru_layer_step_cuda("
    "torch::Tensor x_t,"
    "torch::Tensor h_fwd_prev,"
    "torch::Tensor h_bwd_prev,"
    "torch::Tensor w_ih_fwd,"
    "torch::Tensor b_ih_fwd,"
    "torch::Tensor w_hh_fwd,"
    "torch::Tensor b_hh_fwd,"
    "torch::Tensor w_ih_bwd,"
    "torch::Tensor b_ih_bwd,"
    "torch::Tensor w_hh_bwd,"
    "torch::Tensor b_hh_bwd"
    ");"
)

# Compile the inline CUDA code
gru_ops = load_inline(
    name="gru_ops",
    cpp_sources=gru_cpp_source,
    cuda_sources=gru_source,
    functions=["gru_layer_step_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        """
        :param input_size: The number of expected features in the input x
        :param hidden_size: The number of features in the hidden state h
        :param num_layers: Number of recurrent layers (default: 1)
        :param bias: If False, then the layer does not use bias weights b_ih and b_hh (default: True)
        :param batch_first: If True, then the input and output tensors are provided as (batch, seq, feature) (default: False)
        """
        super(ModelNew, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        
        # Initialize weights and biases for the GRU
        # We need 2 directions (fwd, bwd) per layer
        # Each direction has W_ih, b_ih, W_hh, b_hh
        # Shape of W_ih: [3*hidden_size, input_size]
        # Shape of W_hh: [3*hidden_size, hidden_size]
        
        self.w_ih_fwd = nn.ParameterList([nn.Parameter(torch.randn(3 * hidden_size, input_size)) for _ in range(num_layers)])
        self.b_ih_fwd = nn.ParameterList([nn.Parameter(torch.randn(3 * hidden_size)) for _ in range(num_layers)])
        self.w_hh_fwd = nn.ParameterList([nn.Parameter(torch.randn(3 * hidden_size, hidden_size)) for _ in range(num_layers)])
        self.b_hh_fwd = nn.ParameterList([nn.Parameter(torch.randn(3 * hidden_size)) for _ in range(num_layers)])
        
        self.w_ih_bwd = nn.ParameterList([nn.Parameter(torch.randn(3 * hidden_size, input_size)) for _ in range(num_layers)])
        self.b_ih_bwd = nn.ParameterList([nn.Parameter(torch.randn(3 * hidden_size)) for _ in range(num_layers)])
        self.w_hh_bwd = nn.ParameterList([nn.Parameter(torch.randn(3 * hidden_size, hidden_size)) for _ in range(num_layers)])
        self.b_hh_bwd = nn.ParameterList([nn.Parameter(torch.randn(3 * hidden_size)) for _ in range(num_layers)])

    def forward(self, x, h0):
        """
        :param x: The input tensor
        :param h0: The initial hidden state
        :return: output, h_n
        """
        if self.batch_first:
            # x: [batch, seq_len, input_size] -> [seq_len, batch, input_size]
            x = x.permute(1, 0, 2)
        
        seq_len = x.size(0)
        batch_size = x.size(1)
        
        # h0 shape: [num_layers * 2, batch, hidden_size]
        # We split h0 into forward and backward for each layer
        h_fwd = h0[0::2]  # [num_layers, batch, hidden_size]
        h_bwd = h0[1::2]  # [num_layers, batch, hidden_size]
        
        output_list = []
        
        for t in range(seq_len):
            x_t = x[t]  # [batch, input_size]
            
            h_fwd_next_list = []
            h_bwd_next_list = []
            
            for l in range(self.num_layers):
                # Get weights for layer l
                w_ih_f = self.w_ih_fwd[l]
                b_ih_f = self.b_ih_fwd[l]
                w_hh_f = self.w_hh_fwd[l]
                b_hh_f = self.b_hh_fwd[l]
                
                w_ih_b = self.w_ih_bwd[l]
                b_ih_b = self.b_ih_bwd[l]
                w_hh_b = self.w_hh_bwd[l]
                b_hh_b = self.b_hh_bwd[l]
                
                # Get previous hidden states for this layer
                h_fwd_prev = h_fwd[l]  # [batch, hidden_size]
                h_bwd_prev = h_bwd[l]  # [batch, hidden_size]
                
                # Call custom CUDA kernel
                h_fwd_next, h_bwd_next = gru_ops.gru_layer_step_cuda(
                    x_t, h_fwd_prev, h_bwd_prev,
                    w_ih_f, b_ih_f, w_hh_f, b_hh_f,
                    w_ih_b, b_ih_b, w_hh_b, b_hh_b
                )
                
                h_fwd_next_list.append