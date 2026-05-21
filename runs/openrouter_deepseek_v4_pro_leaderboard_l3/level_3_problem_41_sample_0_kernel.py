import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for bidirectional GRU with multiple layers
gru_cuda_source = """
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
    const float* __restrict__ b_ih,       // input bias: (3*hidden_size)
    const float* __restrict__ b_hh,       // hidden bias: (3*hidden_size)
    float* __restrict__ h_new,            // new hidden: (batch_size, hidden_size)
    int batch_size,
    int input_size,
    int hidden_size
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    // Each thread handles one element of the batch
    // Compute gates: r, z, n
    // We'll compute all gates for this batch element
    
    // Pointers to the specific batch element
    const float* x_b = x + b * input_size;
    const float* h_prev_b = h_prev + b * hidden_size;
    float* h_new_b = h_new + b * hidden_size;

    // Compute gate pre-activations
    // For each gate: W_ih * x + W_hh * h_prev + b_ih + b_hh
    // We'll compute element by element for each hidden unit
    
    for (int i = 0; i < hidden_size; i++) {
        // Reset gate r_i
        float r_i = b_ih[i] + b_hh[i];
        for (int j = 0; j < input_size; j++) {
            r_i += W_ih[i * input_size + j] * x_b[j];
        }
        for (int j = 0; j < hidden_size; j++) {
            r_i += W_hh[i * hidden_size + j] * h_prev_b[j];
        }
        r_i = sigmoid(r_i);

        // Update gate z_i
        float z_i = b_ih[hidden_size + i] + b_hh[hidden_size + i];
        for (int j = 0; j < input_size; j++) {
            z_i += W_ih[(hidden_size + i) * input_size + j] * x_b[j];
        }
        for (int j = 0; j < hidden_size; j++) {
            z_i += W_hh[(hidden_size + i) * hidden_size + j] * h_prev_b[j];
        }
        z_i = sigmoid(z_i);

        // New gate n_i
        float n_i = b_ih[2 * hidden_size + i] + b_hh[2 * hidden_size + i];
        for (int j = 0; j < input_size; j++) {
            n_i += W_ih[(2 * hidden_size + i) * input_size + j] * x_b[j];
        }
        for (int j = 0; j < hidden_size; j++) {
            n_i += W_hh[(2 * hidden_size + i) * hidden_size + j] * (r_i * h_prev_b[j]);
        }
        n_i = tanhf(n_i);

        // New hidden state
        h_new_b[i] = (1.0f - z_i) * n_i + z_i * h_prev_b[i];
    }
}

// Kernel for processing a full sequence with multiple layers and bidirectional
__global__ void bidirectional_gru_kernel(
    const float* __restrict__ x,          // input: (seq_len, batch_size, input_size) or (batch_size, seq_len, input_size) if batch_first
    const float* __restrict__ h0,         // initial hidden: (num_layers*2, batch_size, hidden_size)
    const float* __restrict__ W_ih_l0_rev, // weights for layer 0 reverse direction
    const float* __restrict__ W_hh_l0_rev,
    const float* __restrict__ b_ih_l0_rev,
    const float* __restrict__ b_hh_l0_rev,
    float* __restrict__ output,           // output: (seq_len, batch_size, 2*hidden_size)
    float* __restrict__ h_n,              // final hidden: (num_layers*2, batch_size, hidden_size)
    int seq_len,
    int batch_size,
    int input_size,
    int hidden_size,
    int num_layers,
    bool batch_first
) {
    // This kernel processes the entire GRU stack
    // We'll use a 2D grid: (batch_size, seq_len) for parallelism
    int b = blockIdx.x;
    int t = blockIdx.y;
    
    if (b >= batch_size || t >= seq_len) return;

    // For simplicity, we'll process each time step and layer sequentially within the kernel
    // This is not optimal but demonstrates the concept
    // A more optimized version would use shared memory and better parallelism
    
    // We'll allocate shared memory for intermediate hidden states
    extern __shared__ float shared_mem[];
    // Layout: [2 * num_layers * hidden_size] for current hidden states
    float* h_forward = shared_mem;  // (num_layers, hidden_size) for forward
    float* h_backward = shared_mem + num_layers * hidden_size;  // (num_layers, hidden_size) for backward
    
    // Initialize hidden states from h0
    for (int l = 0; l < num_layers; l++) {
        for (int i = 0; i < hidden_size; i++) {
            h_forward[l * hidden_size + i] = h0[(l * 2) * batch_size * hidden_size + b * hidden_size + i];
            h_backward[l * hidden_size + i] = h0[(l * 2 + 1) * batch_size * hidden_size + b * hidden_size + i];
        }
    }
    
    // Process forward direction for all layers at time step t
    // Input to first layer is x[t]
    // For subsequent layers, input is the hidden state of previous layer at time t
    
    // We need to compute forward hidden for all layers at time t
    // This requires sequential processing across layers
    
    // For simplicity, we'll just compute for the current time step
    // In a real implementation, we'd need to process the whole sequence
    
    // Since we can't easily do sequential time processing in parallel,
    // we'll use a different approach: each thread block handles one batch element
    // and processes the entire sequence sequentially.
    
    // Actually, let's redesign: use one block per batch element, threads per hidden unit
    // and process time sequentially within the block.
    
    // But that would require a different kernel launch configuration.
    // For this example, we'll create a simpler kernel that processes one time step.
    
    // This is getting complex. Let's create a more practical kernel.
}

// Simpler approach: Fused GRU cell that processes one time step for all layers and directions
__global__ void fused_gru_step_kernel(
    const float* __restrict__ x_t,        // input at time t: (batch_size, input_size)
    const float* __restrict__ h_prev,     // previous hidden: (num_layers*2, batch_size, hidden_size)
    const float* __restrict__ weights_ih, // all input-hidden weights: (num_layers*2, 3*hidden_size, input_size)
    const float* __restrict__ weights_hh, // all hidden-hidden weights: (num_layers*2, 3*hidden_size, hidden_size)
    const float* __restrict__ bias_ih,    // all input biases: (num_layers*2, 3*hidden_size)
    const float* __restrict__ bias_hh,    // all hidden biases: (num_layers*2, 3*hidden_size)
    float* __restrict__ h_new,            // new hidden: (num_layers*2, batch_size, hidden_size)
    float* __restrict__ output_t,         // output at time t: (batch_size, num_directions*hidden_size)
    int batch_size,
    int input_size,
    int hidden_size,
    int num_layers,
    int num_directions
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    // For each direction and layer
    for (int d = 0; d < num_directions; d++) {
        for (int l = 0; l < num_layers; l++) {
            int layer_idx = d * num_layers + l;
            
            // Determine input to this layer
            const float* layer_input;
            int layer_input_size;
            if (l == 0) {
                layer_input = x_t;
                layer_input_size = input_size;
            } else {
                // Input is the hidden state of previous layer (same direction)
                int prev_layer_idx = d * num_layers + (l - 1);
                layer_input = h_prev + prev_layer_idx * batch_size * hidden_size + b * hidden_size;
                layer_input_size = hidden_size;
            }
            
            // Previous hidden for this layer
            const float* h_prev_layer = h_prev + layer_idx * batch_size * hidden_size + b * hidden_size;
            
            // Weights for this layer
            const float* W_ih = weights_ih + layer_idx * (3 * hidden_size) * (l == 0 ? input_size : hidden_size);
            const float* W_hh = weights_hh + layer_idx * (3 * hidden_size) * hidden_size;
            const float* b_ih_layer = bias_ih + layer_idx * (3 * hidden_size);
            const float* b_hh_layer = bias_hh + layer_idx * (3 * hidden_size);
            
            // Output hidden for this layer
            float* h_new_layer = h_new + layer_idx * batch_size * hidden_size + b * hidden_size;
            
            // Compute GRU cell
            for (int i = 0; i < hidden_size; i++) {
                // Reset gate
                float r_i = b_ih_layer[i] + b_hh_layer[i];
                for (int j = 0; j < layer_input_size; j++) {
                    r_i += W_ih[i * layer_input_size + j] * layer_input[j];
                }
                for (int j = 0; j < hidden_size; j++) {
                    r_i += W_hh[i * hidden_size + j] * h_prev_layer[j];
                }
                r_i = 1.0f / (1.0f + expf(-r_i));
                
                // Update gate
                float z_i = b_ih_layer[hidden_size + i] + b_hh_layer[hidden_size + i];
                for (int j = 0; j < layer_input_size; j++) {
                    z_i += W_ih[(hidden_size + i) * layer_input_size + j] * layer_input[j];
                }
                for (int j = 0; j < hidden_size; j++) {
                    z_i += W_hh[(hidden_size + i) * hidden_size + j] * h_prev_layer[j];
                }
                z_i = 1.0f / (1.0f + expf(-z_i));
                
                // New gate
                float n_i = b_ih_layer[2 * hidden_size + i] + b_hh_layer[2 * hidden_size + i];
                for (int j = 0; j < layer_input_size; j++) {
                    n_i += W_ih[(2 * hidden_size + i) * layer_input_size + j] * layer_input[j];
                }
                for (int j = 0; j < hidden_size; j++) {
                    n_i += W_hh[(2 * hidden_size + i) * hidden_size + j] * (r_i * h_prev_layer[j]);
                }
                n_i = tanhf(n_i);
                
                h_new_layer[i] = (1.0f - z_i) * n_i + z_i * h_prev_layer[i];
            }
        }
    }
    
    // Copy last layer hidden states to output
    // Forward direction: last layer is num_layers-1
    // Backward direction: last layer is 2*num_layers-1
    float* out = output_t + b * (num_directions * hidden_size);
    for (int i = 0; i < hidden_size; i++) {
        out[i] = h_new[(num_layers - 1) * batch_size * hidden_size + b * hidden_size + i];
        if (num_directions == 2) {
            out[hidden_size + i] = h_new[(2 * num_layers - 1) * batch_size * hidden_size + b * hidden_size + i];
        }
    }
}

// Host function that processes the entire sequence
torch::Tensor gru_forward_cuda(
    torch::Tensor x,
    torch::Tensor h0,
    torch::Tensor weights_ih,
    torch::Tensor weights_hh,
    torch::Tensor bias_ih,
    torch::Tensor bias_hh,
    int num_layers,
    bool batch_first
) {
    // Assume input is (seq_len, batch_size, input_size) if not batch_first
    // We'll handle batch_first=False for simplicity
    int seq_len = x.size(0);
    int batch_size = x.size(1);
    int input_size = x.size(2);
    int hidden_size = h0.size(2);
    int num_directions = 2; // bidirectional
    
    auto output = torch::zeros({seq_len, batch_size, num_directions * hidden_size}, x.options());
    auto h_n = torch::zeros({num_layers * num_directions, batch_size, hidden_size}, x.options());
    
    // Initialize hidden state
    auto h_prev = h0.clone();
    
    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;
    
    // Process forward direction (time steps 0 to seq_len-1)
    for (int t = 0; t < seq_len; t++) {
        auto x_t = x[t]; // (batch_size, input_size)
        auto h_new = torch::zeros_like(h_prev);
        auto output_t = output[t]; // (batch_size, num_directions*hidden_size)
        
        fused_gru_step_kernel<<<num_blocks, block_size>>>(
            x_t.data_ptr<float>(),
            h_prev.data_ptr<float>(),
            weights_ih.data_ptr<float>(),
            weights_hh.data_ptr<float>(),
            bias_ih.data_ptr<float>(),
            bias_hh.data_ptr<float>(),
            h_new.data_ptr<float>(),
            output_t.data_ptr<float>(),
            batch_size,
            input_size,
            hidden_size,
            num_layers,
            num_directions
        );
        
        h_prev = h_new;
    }
    
    // For backward direction, we need to process in reverse
    // This requires a separate pass or a different kernel
    // For simplicity, we'll just return the forward pass result
    // In a complete implementation, we'd process both directions
    
    // Actually, the kernel above processes both directions but uses the same time step
    // For bidirectional, we need to process forward and backward separately
    // Let's create a proper implementation
    
    // We'll process forward direction first
    auto h_forward = h0.narrow(0, 0, num_layers).clone(); // first num_layers are forward
    auto h_backward = h0.narrow(0, num_layers, num_layers).clone(); // last num_layers are backward
    
    // Forward pass
    for (int t = 0; t < seq_len; t++) {
        auto x_t = x[t];
        auto h_new_forward = torch::zeros_like(h_forward);
        auto output_t = output[t].narrow(1, 0, hidden_size); // forward part of output
        
        // Process forward layers
        // We'll use a simpler kernel for forward only
        // For now, we'll just copy the logic
        
        // This is getting too complex for a single kernel. Let's use a different approach.
    }
    
    return output;
}

// Let's create a more practical implementation using a single kernel per time step
// that handles both directions properly.

__global__ void gru_bidirectional_step_kernel(
    const float* __restrict__ x_t,
    const float* __restrict__ h_forward_prev,
    const float* __restrict__ h_backward_prev,
    const float* __restrict__ W_ih_forward,
    const float* __restrict__ W_hh_forward,
    const float* __restrict__ b_ih_forward,
    const float* __restrict__ b_hh_forward,
    const float* __restrict__ W_ih_backward,
    const float* __restrict__ W_hh_backward,
    const float* __restrict__ b_ih_backward,
    const float* __restrict__ b_hh_backward,
    float* __restrict__ h_forward_new,
    float* __restrict__ h_backward_new,
    float* __restrict__ output_t,
    int batch_size,
    int input_size,
    int hidden_size,
    int num_layers
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    // Process forward layers
    const float* layer_input = x_t + b * input_size;
    int layer_input_size = input_size;
    float* h_forward_prev_b = (float*)h_forward_prev + b * hidden_size;
    float* h_forward_new_b = (float*)h_forward_new + b * hidden_size;
    
    for (int l = 0; l < num_layers; l++) {
        const float* W_ih = W_ih_forward + l * (3 * hidden_size) * (l == 0 ? input_size : hidden_size);
        const float* W_hh = W_hh_forward + l * (3 * hidden_size) * hidden_size;
        const float* b_ih = b_ih_forward + l * (3 * hidden_size);
        const float* b_hh = b_hh_forward + l * (3 * hidden_size);
        
        float* h_new = h_forward_new_b + l * batch_size * hidden_size;
        const float* h_prev = h_forward_prev_b + l * batch_size * hidden_size;
        
        for (int i = 0; i < hidden_size; i++) {
            float r_i = b_ih[i] + b_hh[i];
            for (int j = 0; j < layer_input_size; j++) {
                r_i += W_ih[i * layer_input_size + j] * layer_input[j];
            }
            for (int j = 0; j < hidden_size; j++) {
                r_i += W_hh[i * hidden_size + j] * h_prev[j];
            }
            r_i = 1.0f / (1.0f + expf(-r_i));
            
            float z_i = b_ih[hidden_size + i] + b_hh[hidden_size + i];
            for (int j = 0; j < layer_input_size; j++) {
                z_i += W_ih[(hidden_size + i) * layer_input_size + j] * layer_input[j];
            }
            for (int j = 0; j < hidden_size; j++) {
                z_i += W_hh[(hidden_size + i) * hidden_size + j] * h_prev[j];
            }
            z_i = 1.0f / (1.0f + expf(-z_i));
            
            float n_i = b_ih[2 * hidden_size + i] + b_hh[2 * hidden_size + i];
            for (int j = 0; j < layer_input_size; j++) {
                n_i += W_ih[(2 * hidden_size + i) * layer_input_size + j] * layer_input[j];
            }
            for (int j = 0; j < hidden_size; j++) {
                n_i += W_hh[(2 * hidden_size + i) * hidden_size + j] * (r_i * h_prev[j]);
            }
            n_i = tanhf(n_i);
            
            h_new[i] = (1.0f - z_i) * n_i + z_i * h_prev[i];
        }
        
        // Next layer input is this layer's hidden state
        layer_input = h_new;
        layer_input_size = hidden_size;
    }
    
    // Process backward layers (similar but with backward weights)
    // For backward, we process the same way but with different weights
    // The backward direction will be processed in reverse time order
    
    // Copy last forward layer to output (forward part)
    float* out = output_t + b * (2 * hidden_size);
    for (int i = 0; i < hidden_size; i++) {
        out[i] = h_forward_new_b[(num_layers - 1) * batch_size * hidden_size + i];
    }
}

// Host function that orchestrates the full bidirectional GRU
std::vector<torch::Tensor> bidirectional_gru_cuda(
    torch::Tensor x,
    torch::Tensor h0,
    torch::Tensor weights_ih,  // (num_layers*2, 3*hidden_size, input_size) for layer 0, then (num_layers*2, 3*hidden_size, hidden_size) for others
    torch::Tensor weights_hh,  // (num_layers*2, 3*hidden_size, hidden_size)
    torch::Tensor bias_ih,     // (num_layers*2, 3*hidden_size)
    torch::Tensor bias_hh,     // (num_layers*2, 3*hidden_size)
    int num_layers,
    bool batch_first
) {
    int seq_len, batch_size, input_size;
    if (batch_first) {
        batch_size = x.size(0);
        seq_len = x.size(1);
        input_size = x.size(2);
    } else {
        seq_len = x.size(0);
        batch_size = x.size(1);
        input_size = x.size(2);
    }
    int hidden_size = h0.size(2);
    int num_directions = 2;
    
    auto output = torch::zeros({seq_len, batch_size, num_directions * hidden_size}, x.options());
    auto h_n = torch::zeros({num_layers * num_directions, batch_size, hidden_size}, x.options());
    
    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;
    
    // Split h0 into forward and backward
    auto h_forward = h0.narrow(0, 0, num_layers).clone();
    auto h_backward = h0.narrow(0, num_layers, num_layers).clone();
    
    // Forward pass
    for (int t = 0; t < seq_len; t++) {
        auto x_t = batch_first ? x.select(1, t) : x[t];
        auto h_forward_new = torch::zeros_like(h_forward);
        auto output_t = output[t];
        
        // Process forward layers
        // We'll use a separate kernel for forward only
        // For simplicity, we'll just call the kernel for each layer
        // In a real implementation, we'd fuse layers
        
        // This is placeholder; we need to implement the actual forward pass
    }
    
    // Backward pass (reverse time)
    for (int t = seq_len - 1; t >= 0; t--) {
        auto x_t = batch_first ? x.select(1, t) : x[t];
        // Process backward layers
    }
    
    return {output, h_n};
}
"""

# Actually, let's simplify and create a more practical implementation
# We'll create a kernel that processes one GRU cell and use it in a loop
# This is not fully optimized but demonstrates the concept

gru_cuda_source_simple = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void gru_cell_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ h_prev,
    const float* __restrict__ W_ih,
    const float* __restrict__ W_hh,
    const float* __restrict__ b_ih,
    const float* __restrict__ b_hh,
    float* __restrict__ h_new,
    int batch_size,
    int input_size,
    int hidden_size
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    const float* x_b = x + b * input_size;
    const float* h_prev_b = h_prev + b * hidden_size;
    float* h_new_b = h_new + b * hidden_size;

    for (int i = 0; i < hidden_size; i++) {
        // Reset gate
        float r_i = b_ih[i] + b_hh[i];
        for (int j = 0; j < input_size; j++) {
            r_i += W_ih[i * input_size + j] * x_b[j];
        }
        for (int j = 0; j < hidden_size; j++) {
            r_i += W_hh[i * hidden_size + j] * h_prev_b[j];
        }
        r_i = 1.0f / (1.0f + expf(-r_i));

        // Update gate
        float z_i = b_ih[hidden_size + i] + b_hh[hidden_size + i];
        for (int j = 0; j < input_size; j++) {
            z_i += W_ih[(hidden_size + i) * input_size + j] * x_b[j];
        }
        for (int j = 0; j < hidden_size; j++) {
            z_i += W_hh[(hidden_size + i) * hidden_size + j] * h_prev_b[j];
        }
        z_i = 1.0f / (1.0f + expf(-z_i));

        // New gate
        float n_i = b_ih[2 * hidden_size + i] + b_hh[2 * hidden_size + i];
        for (int j = 0; j < input_size; j++) {
            n_i += W_ih[(2 * hidden_size + i) * input_size + j] * x_b[j];
        }
        for (int j = 0; j < hidden_size; j++) {
            n_i += W_hh[(2 * hidden_size + i) * hidden_size + j] * (r_i * h_prev_b[j]);
        }
        n_i = tanhf(n_i);

        h_new_b[i] = (1.0f - z_i) * n_i + z_i * h_prev_b[i];
    }
}

torch::Tensor gru_cell_forward_cuda(
    torch::Tensor x,
    torch::Tensor h_prev,
    torch::Tensor W_ih,
    torch::Tensor W_hh,
    torch::Tensor b_ih,
    torch::Tensor b_hh
) {
    int batch_size = x.size(0);
    int input_size = x.size(1);
    int hidden_size = h_prev.size(1);
    
    auto h_new = torch::zeros_like(h_prev);
    
    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;
    
    gru_cell_forward_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        h_prev.data_ptr<float>(),
        W_ih.data_ptr<float>(),
        W_hh.data_ptr<float>(),
        b_ih.data_ptr<float>(),
        b_hh.data_ptr<float>(),
        h_new.data_ptr<float>(),
        batch_size,
        input_size,
        hidden_size
    );
    
    return h_new;
}
"""

gru_cpp_source = "torch::Tensor gru_cell_forward_cuda(torch::Tensor x, torch::Tensor h_prev, torch::Tensor W_ih, torch::Tensor W_hh, torch::Tensor b_ih, torch::Tensor b_hh);"

# Compile the custom CUDA kernel
gru_cell = load_inline(
    name="gru_cell",
    cpp_sources=gru_cpp_source,
    cuda_sources=gru_cuda_source_simple,
    functions=["gru_cell_forward_cuda"],
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
        self.num_directions = 2  # bidirectional
        
        # Create GRU parameters manually
        # For each direction and layer, we need W_ih, W_hh, b_ih, b_hh
        # PyTorch GRU stores weights in a specific order:
        # weight_ih_l[k] = W_ir, W_iz, W_in for layer k
        # weight_hh_l[k] = W_hr, W_hz, W_hn
        # For bidirectional: weight_ih_l[k]_reverse, etc.
        
        # We'll create parameters for each layer and direction
        self.weights_ih = nn.ParameterList()
        self.weights_hh = nn.ParameterList()
        self.bias_ih = nn.ParameterList()
        self.bias_hh = nn.ParameterList()
        
        for layer in range(num_layers):
            for direction in range(self.num_directions):
                suffix = '_reverse' if direction == 1 else ''
                layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
                
                w_ih = nn.Parameter(torch.randn(3 * hidden_size, layer_input_size))
                w_hh = nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
                b_ih = nn.Parameter(torch.randn(3 * hidden_size))
                b_hh = nn.Parameter(torch.randn(3 * hidden_size))
                
                self.weights_ih.append(w_ih)
                self.weights_hh.append(w_hh)
                self.bias_ih.append(b_ih)
                self.bias_hh.append(b_hh)
        
        self.gru_cell = gru_cell
        
    def forward(self, x, h0):
        if self.batch_first:
            x = x.transpose(0, 1)  # Convert to (seq_len, batch_size, input_size)
        
        seq_len, batch_size, _ = x.shape
        
        # Initialize output tensor
        output = torch.zeros(seq_len, batch_size, self.hidden_size * self.num_directions, device=x.device)
        h_n = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=x.device)
        
        # Process each direction
        for direction in range(self.num_directions):
            # Initial hidden state for this direction
            h = h0[direction * self.num_layers:(direction + 1) * self.num_layers].clone()
            
            # Time steps
            time_steps = range(seq_len) if direction == 0 else range(seq_len - 1, -1, -1)
            
            for t in time_steps:
                x_t = x[t]  # (batch_size, input_size)
                
                # Process through layers
                for layer in range(self.num_layers):
                    idx = direction * self.num_layers + layer
                    layer_input = x_t if layer == 0 else h[layer - 1]
                    
                    h[layer] = self.gru_cell.gru_cell_forward_cuda(
                        layer_input,
                        h[layer],
                        self.weights_ih[idx],
                        self.weights_hh[idx],
                        self.bias_ih[idx],
                        self.bias_hh[idx]
                    )
                
                # Store output from last layer
                if direction == 0:
                    output[t, :, :self.hidden_size] = h[-1]
                else:
                    output[t, :, self.hidden_size:] = h[-1]
            
            # Store final hidden state
            h_n[direction * self.num_layers:(direction + 1) * self.num_layers] = h
        
        if self.batch_first:
            output = output.transpose(0, 1)
        
        return output, h_n