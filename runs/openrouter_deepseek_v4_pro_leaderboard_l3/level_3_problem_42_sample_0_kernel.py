import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for bidirectional GRU with multiple layers
bidirectional_gru_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Sigmoid function
__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// GRU cell forward for a single direction
__global__ void gru_cell_kernel(
    const float* __restrict__ x,          // input: (batch_size, input_size)
    const float* __restrict__ h_prev,     // previous hidden: (batch_size, hidden_size)
    const float* __restrict__ W_ih,       // input-hidden weights: (3*hidden_size, input_size)
    const float* __restrict__ W_hh,       // hidden-hidden weights: (3*hidden_size, hidden_size)
    const float* __restrict__ b_ih,       // input-hidden bias: (3*hidden_size)
    const float* __restrict__ b_hh,       // hidden-hidden bias: (3*hidden_size)
    float* __restrict__ h_new,            // new hidden: (batch_size, hidden_size)
    int batch_size,
    int input_size,
    int hidden_size
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    // Each thread handles one element of the hidden state for one batch element
    // We'll use a warp-level approach for efficiency, but for simplicity, we process per-element
    // Actually, we need to compute the full hidden state for each batch element.
    // Let's use a block per batch element with threads computing hidden elements.
    // Redesign: block per batch element, threads compute hidden elements.
    // But we need to compute matrix-vector products. Let's use shared memory.
    
    extern __shared__ float shared[];
    float* x_shared = shared;
    float* h_shared = shared + input_size;
    
    // Load input and previous hidden into shared memory
    for (int i = threadIdx.x; i < input_size; i += blockDim.x) {
        x_shared[i] = x[b * input_size + i];
    }
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        h_shared[i] = h_prev[b * hidden_size + i];
    }
    __syncthreads();
    
    // Compute gates: r, z, n
    // Each thread computes one element of the hidden state
    int h_idx = threadIdx.x;
    if (h_idx >= hidden_size) return;
    
    // Compute reset gate r[h_idx]
    float r_sum = 0.0f;
    for (int i = 0; i < input_size; i++) {
        r_sum += W_ih[h_idx * input_size + i] * x_shared[i];
    }
    for (int i = 0; i < hidden_size; i++) {
        r_sum += W_hh[h_idx * hidden_size + i] * h_shared[i];
    }
    r_sum += b_ih[h_idx] + b_hh[h_idx];
    float r = sigmoid(r_sum);
    
    // Compute update gate z[h_idx]
    float z_sum = 0.0f;
    int offset_z = hidden_size;
    for (int i = 0; i < input_size; i++) {
        z_sum += W_ih[(offset_z + h_idx) * input_size + i] * x_shared[i];
    }
    for (int i = 0; i < hidden_size; i++) {
        z_sum += W_hh[(offset_z + h_idx) * hidden_size + i] * h_shared[i];
    }
    z_sum += b_ih[offset_z + h_idx] + b_hh[offset_z + h_idx];
    float z = sigmoid(z_sum);
    
    // Compute new gate n[h_idx]
    float n_sum = 0.0f;
    int offset_n = 2 * hidden_size;
    for (int i = 0; i < input_size; i++) {
        n_sum += W_ih[(offset_n + h_idx) * input_size + i] * x_shared[i];
    }
    for (int i = 0; i < hidden_size; i++) {
        n_sum += W_hh[(offset_n + h_idx) * hidden_size + i] * (r * h_shared[i]);
    }
    n_sum += b_ih[offset_n + h_idx] + b_hh[offset_n + h_idx];
    float n = tanhf(n_sum);
    
    // Compute new hidden state
    h_new[b * hidden_size + h_idx] = (1.0f - z) * n + z * h_shared[h_idx];
}

// Kernel for processing one layer of bidirectional GRU over all time steps
__global__ void bidirectional_gru_layer_kernel(
    const float* __restrict__ input,      // (seq_len, batch_size, input_size) or (batch_size, seq_len, input_size) if batch_first
    const float* __restrict__ h0_forward, // initial hidden forward: (batch_size, hidden_size)
    const float* __restrict__ h0_backward,// initial hidden backward: (batch_size, hidden_size)
    const float* __restrict__ W_ih_f,     // forward input-hidden weights
    const float* __restrict__ W_hh_f,     // forward hidden-hidden weights
    const float* __restrict__ b_ih_f,     // forward input-hidden bias
    const float* __restrict__ b_hh_f,     // forward hidden-hidden bias
    const float* __restrict__ W_ih_b,     // backward input-hidden weights
    const float* __restrict__ W_hh_b,     // backward hidden-hidden weights
    const float* __restrict__ b_ih_b,     // backward input-hidden bias
    const float* __restrict__ b_hh_b,     // backward hidden-hidden bias
    float* __restrict__ output,           // (seq_len, batch_size, 2*hidden_size)
    float* __restrict__ h_n,              // (2, batch_size, hidden_size) for last layer forward and backward
    int seq_len,
    int batch_size,
    int input_size,
    int hidden_size,
    bool batch_first
) {
    // This kernel is complex. We'll process each direction sequentially.
    // For simplicity, we'll launch separate kernels for forward and backward passes.
    // But here we combine them. Let's use a block per batch element and time step?
    // Better: use a kernel that processes one direction at a time, called from host.
    // We'll implement a simpler approach: host calls separate kernels for each direction and layer.
    // For now, we'll just provide the building blocks and orchestrate from Python.
}

// Forward direction kernel for one layer over all time steps
__global__ void gru_forward_layer_kernel(
    const float* __restrict__ input,      // (seq_len, batch_size, input_size)
    const float* __restrict__ h0,         // (batch_size, hidden_size)
    const float* __restrict__ W_ih,       // (3*hidden_size, input_size)
    const float* __restrict__ W_hh,       // (3*hidden_size, hidden_size)
    const float* __restrict__ b_ih,       // (3*hidden_size)
    const float* __restrict__ b_hh,       // (3*hidden_size)
    float* __restrict__ output,           // (seq_len, batch_size, hidden_size)
    float* __restrict__ h_n,              // (batch_size, hidden_size)
    int seq_len,
    int batch_size,
    int input_size,
    int hidden_size
) {
    // Each block handles one batch element, iterating over time steps
    int b = blockIdx.x;
    if (b >= batch_size) return;
    
    extern __shared__ float shared[];
    float* h_prev = shared;
    float* x_t = shared + hidden_size;
    // We need more shared memory for intermediate computations, but we'll reuse.
    
    // Initialize h_prev with h0
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        h_prev[i] = h0[b * hidden_size + i];
    }
    __syncthreads();
    
    for (int t = 0; t < seq_len; t++) {
        // Load input at time t
        for (int i = threadIdx.x; i < input_size; i += blockDim.x) {
            x_t[i] = input[t * batch_size * input_size + b * input_size + i];
        }
        __syncthreads();
        
        // Compute new hidden state
        int h_idx = threadIdx.x;
        if (h_idx < hidden_size) {
            // Reset gate
            float r_sum = 0.0f;
            for (int i = 0; i < input_size; i++) {
                r_sum += W_ih[h_idx * input_size + i] * x_t[i];
            }
            for (int i = 0; i < hidden_size; i++) {
                r_sum += W_hh[h_idx * hidden_size + i] * h_prev[i];
            }
            r_sum += b_ih[h_idx] + b_hh[h_idx];
            float r = sigmoid(r_sum);
            
            // Update gate
            float z_sum = 0.0f;
            int offset_z = hidden_size;
            for (int i = 0; i < input_size; i++) {
                z_sum += W_ih[(offset_z + h_idx) * input_size + i] * x_t[i];
            }
            for (int i = 0; i < hidden_size; i++) {
                z_sum += W_hh[(offset_z + h_idx) * hidden_size + i] * h_prev[i];
            }
            z_sum += b_ih[offset_z + h_idx] + b_hh[offset_z + h_idx];
            float z = sigmoid(z_sum);
            
            // New gate
            float n_sum = 0.0f;
            int offset_n = 2 * hidden_size;
            for (int i = 0; i < input_size; i++) {
                n_sum += W_ih[(offset_n + h_idx) * input_size + i] * x_t[i];
            }
            for (int i = 0; i < hidden_size; i++) {
                n_sum += W_hh[(offset_n + h_idx) * hidden_size + i] * (r * h_prev[i]);
            }
            n_sum += b_ih[offset_n + h_idx] + b_hh[offset_n + h_idx];
            float n = tanhf(n_sum);
            
            float h_new = (1.0f - z) * n + z * h_prev[h_idx];
            
            // Store output
            output[t * batch_size * hidden_size + b * hidden_size + h_idx] = h_new;
            // Update h_prev for next time step (need to sync later)
            // We'll store in shared memory after all threads compute
        }
        __syncthreads();
        // Update h_prev from output
        for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
            h_prev[i] = output[t * batch_size * hidden_size + b * hidden_size + i];
        }
        __syncthreads();
    }
    
    // Store final hidden state
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        h_n[b * hidden_size + i] = h_prev[i];
    }
}

// Backward direction kernel for one layer over all time steps
__global__ void gru_backward_layer_kernel(
    const float* __restrict__ input,      // (seq_len, batch_size, input_size)
    const float* __restrict__ h0,         // (batch_size, hidden_size)
    const float* __restrict__ W_ih,       // (3*hidden_size, input_size)
    const float* __restrict__ W_hh,       // (3*hidden_size, hidden_size)
    const float* __restrict__ b_ih,       // (3*hidden_size)
    const float* __restrict__ b_hh,       // (3*hidden_size)
    float* __restrict__ output,           // (seq_len, batch_size, hidden_size)
    float* __restrict__ h_n,              // (batch_size, hidden_size)
    int seq_len,
    int batch_size,
    int input_size,
    int hidden_size
) {
    int b = blockIdx.x;
    if (b >= batch_size) return;
    
    extern __shared__ float shared[];
    float* h_prev = shared;
    float* x_t = shared + hidden_size;
    
    // Initialize h_prev with h0 (which is the initial hidden for backward direction, i.e., at time seq_len-1)
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        h_prev[i] = h0[b * hidden_size + i];
    }
    __syncthreads();
    
    // Process in reverse order
    for (int t = seq_len - 1; t >= 0; t--) {
        // Load input at time t
        for (int i = threadIdx.x; i < input_size; i += blockDim.x) {
            x_t[i] = input[t * batch_size * input_size + b * input_size + i];
        }
        __syncthreads();
        
        int h_idx = threadIdx.x;
        if (h_idx < hidden_size) {
            // Reset gate
            float r_sum = 0.0f;
            for (int i = 0; i < input_size; i++) {
                r_sum += W_ih[h_idx * input_size + i] * x_t[i];
            }
            for (int i = 0; i < hidden_size; i++) {
                r_sum += W_hh[h_idx * hidden_size + i] * h_prev[i];
            }
            r_sum += b_ih[h_idx] + b_hh[h_idx];
            float r = sigmoid(r_sum);
            
            // Update gate
            float z_sum = 0.0f;
            int offset_z = hidden_size;
            for (int i = 0; i < input_size; i++) {
                z_sum += W_ih[(offset_z + h_idx) * input_size + i] * x_t[i];
            }
            for (int i = 0; i < hidden_size; i++) {
                z_sum += W_hh[(offset_z + h_idx) * hidden_size + i] * h_prev[i];
            }
            z_sum += b_ih[offset_z + h_idx] + b_hh[offset_z + h_idx];
            float z = sigmoid(z_sum);
            
            // New gate
            float n_sum = 0.0f;
            int offset_n = 2 * hidden_size;
            for (int i = 0; i < input_size; i++) {
                n_sum += W_ih[(offset_n + h_idx) * input_size + i] * x_t[i];
            }
            for (int i = 0; i < hidden_size; i++) {
                n_sum += W_hh[(offset_n + h_idx) * hidden_size + i] * (r * h_prev[i]);
            }
            n_sum += b_ih[offset_n + h_idx] + b_hh[offset_n + h_idx];
            float n = tanhf(n_sum);
            
            float h_new = (1.0f - z) * n + z * h_prev[h_idx];
            
            // Store output (backward direction output is stored in reverse order? Actually, we store at time t)
            output[t * batch_size * hidden_size + b * hidden_size + h_idx] = h_new;
        }
        __syncthreads();
        // Update h_prev
        for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
            h_prev[i] = output[t * batch_size * hidden_size + b * hidden_size + i];
        }
        __syncthreads();
    }
    
    // Store final hidden state (which is at time 0 after backward processing)
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        h_n[b * hidden_size + i] = h_prev[i];
    }
}

// Host function to run one layer of bidirectional GRU
torch::Tensor bidirectional_gru_layer_cuda(
    torch::Tensor input,           // (seq_len, batch_size, input_size)
    torch::Tensor h0,              // (2, batch_size, hidden_size) for bidirectional, first is forward, second is backward
    torch::Tensor W_ih_f, torch::Tensor W_hh_f, torch::Tensor b_ih_f, torch::Tensor b_hh_f,
    torch::Tensor W_ih_b, torch::Tensor W_hh_b, torch::Tensor b_ih_b, torch::Tensor b_hh_b,
    int seq_len, int batch_size, int input_size, int hidden_size
) {
    // Allocate output and h_n
    auto output = torch::zeros({seq_len, batch_size, 2 * hidden_size}, input.options());
    auto h_n = torch::zeros({2, batch_size, hidden_size}, input.options());
    
    // Temporary buffers for forward and backward outputs
    auto out_f = torch::zeros({seq_len, batch_size, hidden_size}, input.options());
    auto out_b = torch::zeros({seq_len, batch_size, hidden_size}, input.options());
    
    // Forward pass
    int threads = 256;
    int blocks = batch_size;
    int shared_mem_size = (hidden_size + input_size) * sizeof(float);
    
    gru_forward_layer_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        h0[0].data_ptr<float>(),  // forward initial hidden
        W_ih_f.data_ptr<float>(),
        W_hh_f.data_ptr<float>(),
        b_ih_f.data_ptr<float>(),
        b_hh_f.data_ptr<float>(),
        out_f.data_ptr<float>(),
        h_n[0].data_ptr<float>(),
        seq_len, batch_size, input_size, hidden_size
    );
    
    // Backward pass
    gru_backward_layer_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        h0[1].data_ptr<float>(),  // backward initial hidden
        W_ih_b.data_ptr<float>(),
        W_hh_b.data_ptr<float>(),
        b_ih_b.data_ptr<float>(),
        b_hh_b.data_ptr<float>(),
        out_b.data_ptr<float>(),
        h_n[1].data_ptr<float>(),
        seq_len, batch_size, input_size, hidden_size
    );
    
    // Combine forward and backward outputs: concatenate along last dimension
    // output[:, :, :hidden_size] = out_f, output[:, :, hidden_size:] = out_b
    // We'll do this with a simple kernel or use tensor operations. For simplicity, use a kernel.
    // But we can also do it with a custom kernel or just use PyTorch's cat. Since we're in CUDA, let's write a small kernel.
    // Actually, we can return the two tensors and combine in Python. But we want a single output.
    // Let's write a combine kernel.
    
    // For now, we'll combine in Python after calling this function. We'll modify the host function to return both.
    // But the function signature expects a single output. Let's just return output and h_n.
    // We'll combine in a separate step.
    
    // Combine
    dim3 combine_blocks((seq_len * batch_size * hidden_size + 255) / 256);
    // We'll do this in Python for simplicity.
    
    // Actually, let's just return the two parts and combine in Python.
    // We'll create a new function that returns the combined output.
    
    // For now, we'll just return the output tensor (which is zeros) and fill it in Python.
    // Better: write a combine kernel.
    
    // Let's just use a simple approach: copy data from out_f and out_b to output.
    // We'll do this with cudaMemcpy or a kernel. Let's write a small kernel.
    
    // For simplicity, we'll do the combination in Python using tensor operations.
    // But since we want a pure CUDA solution, we'll write a combine kernel.
    
    // We'll return the output and h_n after combining.
    // Let's write a combine kernel inline.
    
    // Actually, we can just use torch::cat in C++ but that requires including headers. Let's just do it with a kernel.
    
    // We'll modify the function to return a tuple.
    return output; // placeholder, we'll fix later
}

// Combine kernel
__global__ void combine_forward_backward_kernel(
    const float* __restrict__ out_f,
    const float* __restrict__ out_b,
    float* __restrict__ output,
    int seq_len,
    int batch_size,
    int hidden_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * batch_size * hidden_size;
    if (idx < total) {
        // output has shape (seq_len, batch_size, 2*hidden_size)
        // We need to map linear index to (t, b, h)
        int h = idx % hidden_size;
        int b = (idx / hidden_size) % batch_size;
        int t = idx / (hidden_size * batch_size);
        
        // Forward part
        output[t * batch_size * 2 * hidden_size + b * 2 * hidden_size + h] = out_f[idx];
        // Backward part
        output[t * batch_size * 2 * hidden_size + b * 2 * hidden_size + hidden_size + h] = out_b[idx];
    }
}

// Full bidirectional GRU function that handles multiple layers
std::vector<torch::Tensor> bidirectional_gru_cuda(
    torch::Tensor input,           // (seq_len, batch_size, input_size)
    torch::Tensor h0,              // (num_layers*2, batch_size, hidden_size)
    std::vector<torch::Tensor> weights, // flat list of weights for all layers
    int num_layers,
    int seq_len,
    int batch_size,
    int input_size,
    int hidden_size
) {
    // weights order: for each layer: W_ih_f, W_hh_f, b_ih_f, b_hh_f, W_ih_b, W_hh_b, b_ih_b, b_hh_b
    // h0 shape: (num_layers*2, batch_size, hidden_size)
    // We'll process layer by layer
    
    auto x = input;
    auto h_n = torch::zeros({num_layers * 2, batch_size, hidden_size}, input.options());
    
    for (int l = 0; l < num_layers; l++) {
        int idx = l * 8;
        auto W_ih_f = weights[idx];
        auto W_hh_f = weights[idx + 1];
        auto b_ih_f = weights[idx + 2];
        auto b_hh_f = weights[idx + 3];
        auto W_ih_b = weights[idx + 4];
        auto W_hh_b = weights[idx + 5];
        auto b_ih_b = weights[idx + 6];
        auto b_hh_b = weights[idx + 7];
        
        // Get initial hidden for this layer
        auto h0_f = h0[l * 2];     // forward
        auto h0_b = h0[l * 2 + 1]; // backward
        
        int current_input_size = (l == 0) ? input_size : 2 * hidden_size;
        
        // Allocate output for this layer
        auto output = torch::zeros({seq_len, batch_size, 2 * hidden_size}, input.options());
        auto out_f = torch::zeros({seq_len, batch_size, hidden_size}, input.options());
        auto out_b = torch::zeros({seq_len, batch_size, hidden_size}, input.options());
        
        int threads = 256;
        int blocks = batch_size;
        int shared_mem_size = (hidden_size + current_input_size) * sizeof(float);
        
        // Forward pass
        gru_forward_layer_kernel<<<blocks, threads, shared_mem_size>>>(
            x.data_ptr<float>(),
            h0_f.data_ptr<float>(),
            W_ih_f.data_ptr<float>(),
            W_hh_f.data_ptr<float>(),
            b_ih_f.data_ptr<float>(),
            b_hh_f.data_ptr<float>(),
            out_f.data_ptr<float>(),
            h_n[l * 2].data_ptr<float>(),  // store forward h_n
            seq_len, batch_size, current_input_size, hidden_size
        );
        
        // Backward pass
        gru_backward_layer_kernel<<<blocks, threads, shared_mem_size>>>(
            x.data_ptr<float>(),
            h0_b.data_ptr<float>(),
            W_ih_b.data_ptr<float>(),
            W_hh_b.data_ptr<float>(),
            b_ih_b.data_ptr<float>(),
            b_hh_b.data_ptr<float>(),
            out_b.data_ptr<float>(),
            h_n[l * 2 + 1].data_ptr<float>(),  // store backward h_n
            seq_len, batch_size, current_input_size, hidden_size
        );
        
        // Combine forward and backward outputs
        int total = seq_len * batch_size * hidden_size;
        int combine_threads = 256;
        int combine_blocks = (total + combine_threads - 1) / combine_threads;
        combine_forward_backward_kernel<<<combine_blocks, combine_threads>>>(
            out_f.data_ptr<float>(),
            out_b.data_ptr<float>(),
            output.data_ptr<float>(),
            seq_len, batch_size, hidden_size
        );
        
        // Set x for next layer
        x = output;
    }
    
    return {x, h_n};
}
"""

bidirectional_gru_cpp_source = """
std::vector<torch::Tensor> bidirectional_gru_cuda(
    torch::Tensor input,
    torch::Tensor h0,
    std::vector<torch::Tensor> weights,
    int num_layers,
    int seq_len,
    int batch_size,
    int input_size,
    int hidden_size);
"""

# Compile the inline CUDA code
bidirectional_gru = load_inline(
    name="bidirectional_gru",
    cpp_sources=bidirectional_gru_cpp_source,
    cuda_sources=bidirectional_gru_source,
    functions=["bidirectional_gru_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        
        # We'll use the original GRU to get the weights, then extract them for our custom kernel
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bias, batch_first, dropout=0, bidirectional=True)
        self.bidirectional_gru = bidirectional_gru
        
    def forward(self, x, h0):
        # Ensure input is in (seq_len, batch_size, input_size) format
        if self.batch_first:
            x = x.transpose(0, 1)  # (batch, seq, feature) -> (seq, batch, feature)
        
        seq_len, batch_size, _ = x.shape
        
        # Extract weights from the GRU module
        weights = []
        for layer in range(self.num_layers):
            # Forward direction weights
            # weight_ih_l[k] shape: (3*hidden_size, input_size)
            # weight_hh_l[k] shape: (3*hidden_size, hidden_size)
            # bias_ih_l[k] shape: (3*hidden_size)
            # bias_hh_l[k] shape: (3*hidden_size)
            # For bidirectional, we have reverse as well
            # The naming convention: weight_ih_l{layer}, weight_hh_l{layer}, etc.
            # For bidirectional: weight_ih_l{layer}_reverse, etc.
            
            # Get forward weights
            w_ih_f = getattr(self.gru, f'weight_ih_l{layer}')
            w_hh_f = getattr(self.gru, f'weight_hh_l{layer}')
            b_ih_f = getattr(self.gru, f'bias_ih_l{layer}') if self.bias else torch.zeros(3 * self.hidden_size, device=x.device)
            b_hh_f = getattr(self.gru, f'bias_hh_l{layer}') if self.bias else torch.zeros(3 * self.hidden_size, device=x.device)
            
            # Get backward weights
            w_ih_b = getattr(self.gru, f'weight_ih_l{layer}_reverse')
            w_hh_b = getattr(self.gru, f'weight_hh_l{layer}_reverse')
            b_ih_b = getattr(self.gru, f'bias_ih_l{layer}_reverse') if self.bias else torch.zeros(3 * self.hidden_size, device=x.device)
            b_hh_b = getattr(self.gru, f'bias_hh_l{layer}_reverse') if self.bias else torch.zeros(3 * self.hidden_size, device=x.device)
            
            weights.extend([w_ih_f, w_hh_f, b_ih_f, b_hh_f, w_ih_b, w_hh_b, b_ih_b, b_hh_b])
        
        # Call custom CUDA kernel
        output, h_n = self.bidirectional_gru.bidirectional_gru_cuda(
            x, h0, weights, self.num_layers, seq_len, batch_size, 
            self.input_size, self.hidden_size
        )
        
        # If batch_first, transpose output back
        if self.batch_first:
            output = output.transpose(0, 1)
        
        return h_n