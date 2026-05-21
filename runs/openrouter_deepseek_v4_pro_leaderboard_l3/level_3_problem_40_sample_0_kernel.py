import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for GRU forward pass
gru_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// GRU cell forward kernel
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
    int b = blockIdx.x;  // batch index
    int h = threadIdx.x; // hidden dimension index
    
    if (b >= batch_size || h >= hidden_size) return;
    
    // Compute gates: r, z, n
    // Each gate has hidden_size elements
    // W_ih layout: [W_ir | W_iz | W_in] stacked vertically, each (hidden_size, input_size)
    // W_hh layout: [W_hr | W_hz | W_hn] stacked vertically, each (hidden_size, hidden_size)
    
    // Compute reset gate r
    float r_sum = 0.0f;
    for (int i = 0; i < input_size; i++) {
        r_sum += W_ih[h * input_size + i] * x[b * input_size + i];
    }
    for (int i = 0; i < hidden_size; i++) {
        r_sum += W_hh[h * hidden_size + i] * h_prev[b * hidden_size + i];
    }
    r_sum += b_ih[h] + b_hh[h];
    float r = 1.0f / (1.0f + expf(-r_sum));  // sigmoid
    
    // Compute update gate z
    float z_sum = 0.0f;
    int z_offset = hidden_size;
    for (int i = 0; i < input_size; i++) {
        z_sum += W_ih[(z_offset + h) * input_size + i] * x[b * input_size + i];
    }
    for (int i = 0; i < hidden_size; i++) {
        z_sum += W_hh[(z_offset + h) * hidden_size + i] * h_prev[b * hidden_size + i];
    }
    z_sum += b_ih[z_offset + h] + b_hh[z_offset + h];
    float z = 1.0f / (1.0f + expf(-z_sum));  // sigmoid
    
    // Compute new gate n
    float n_sum = 0.0f;
    int n_offset = 2 * hidden_size;
    for (int i = 0; i < input_size; i++) {
        n_sum += W_ih[(n_offset + h) * input_size + i] * x[b * input_size + i];
    }
    for (int i = 0; i < hidden_size; i++) {
        n_sum += W_hh[(n_offset + h) * hidden_size + i] * (r * h_prev[b * hidden_size + i]);
    }
    n_sum += b_ih[n_offset + h] + b_hh[n_offset + h];
    float n = tanhf(n_sum);
    
    // Compute new hidden state
    h_new[b * hidden_size + h] = (1.0f - z) * n + z * h_prev[b * hidden_size + h];
}

// Multi-layer GRU forward pass
torch::Tensor gru_forward_cuda(
    torch::Tensor x,           // (seq_len, batch_size, input_size)
    torch::Tensor h0,          // (num_layers, batch_size, hidden_size)
    torch::Tensor W_ih,        // (num_layers, 3*hidden_size, input_size)
    torch::Tensor W_hh,        // (num_layers, 3*hidden_size, hidden_size)
    torch::Tensor b_ih,        // (num_layers, 3*hidden_size)
    torch::Tensor b_hh,        // (num_layers, 3*hidden_size)
    int num_layers,
    int seq_len,
    int batch_size,
    int input_size,
    int hidden_size
) {
    // Allocate output hidden states for all time steps (we only need the last one, but compute all)
    auto h_out = torch::zeros({num_layers, batch_size, hidden_size}, x.options());
    
    // Temporary buffers for layer inputs/outputs
    auto layer_input = x.clone();  // (seq_len, batch_size, input_size) for first layer
    auto layer_output = torch::zeros({seq_len, batch_size, hidden_size}, x.options());
    
    const int block_size = 256;  // threads per block (one per hidden dimension)
    
    for (int layer = 0; layer < num_layers; layer++) {
        // Get weights and biases for this layer
        auto W_ih_l = W_ih[layer];  // (3*hidden_size, input_size)
        auto W_hh_l = W_hh[layer];  // (3*hidden_size, hidden_size)
        auto b_ih_l = b_ih[layer];  // (3*hidden_size)
        auto b_hh_l = b_hh[layer];  // (3*hidden_size)
        
        // Initial hidden state for this layer
        auto h_prev = h0[layer];  // (batch_size, hidden_size)
        
        // Process sequence
        for (int t = 0; t < seq_len; t++) {
            auto x_t = layer_input[t];  // (batch_size, input_size) or (batch_size, hidden_size) for layers > 0
            
            // Launch kernel: one block per batch element, threads per hidden dimension
            int num_blocks = batch_size;
            int threads_per_block = min(hidden_size, block_size);
            
            gru_cell_kernel<<<num_blocks, threads_per_block>>>(
                x_t.data_ptr<float>(),
                h_prev.data_ptr<float>(),
                W_ih_l.data_ptr<float>(),
                W_hh_l.data_ptr<float>(),
                b_ih_l.data_ptr<float>(),
                b_hh_l.data_ptr<float>(),
                layer_output[t].data_ptr<float>(),
                batch_size,
                (layer == 0) ? input_size : hidden_size,
                hidden_size
            );
            
            // Update h_prev for next time step
            h_prev = layer_output[t];
        }
        
        // Store final hidden state for this layer
        h_out[layer] = h_prev;
        
        // Set input for next layer
        if (layer < num_layers - 1) {
            layer_input = layer_output.clone();
        }
    }
    
    cudaDeviceSynchronize();
    return h_out;  // (num_layers, batch_size, hidden_size)
}
"""

gru_cpp_source = """
torch::Tensor gru_forward_cuda(
    torch::Tensor x,
    torch::Tensor h0,
    torch::Tensor W_ih,
    torch::Tensor W_hh,
    torch::Tensor b_ih,
    torch::Tensor b_hh,
    int num_layers,
    int seq_len,
    int batch_size,
    int input_size,
    int hidden_size
);
"""

# Compile the custom CUDA kernel
gru_custom = load_inline(
    name="gru_custom",
    cpp_sources=gru_cpp_source,
    cuda_sources=gru_cuda_source,
    functions=["gru_forward_cuda"],
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
        
        # Create standard GRU to get weights, but we'll use custom forward
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bias, batch_first, dropout=0, bidirectional=False)
        self.gru_custom = gru_custom
        
    def forward(self, x, h0):
        # Ensure inputs are on the same device as the model
        if self.batch_first:
            x = x.transpose(0, 1)  # Convert to (seq_len, batch_size, input_size)
        
        seq_len, batch_size, _ = x.shape
        
        # Extract weights from the standard GRU
        # GRU weights are stored as a single tensor: (num_layers * 3 * hidden_size, input_size) for W_ih
        # and (num_layers * 3 * hidden_size, hidden_size) for W_hh
        # We need to reshape them per layer
        
        # Get all parameters
        param_dict = dict(self.gru.named_parameters())
        
        # Reshape weight_ih_l[k] and weight_hh_l[k] for each layer
        W_ih_list = []
        W_hh_list = []
        b_ih_list = []
        b_hh_list = []
        
        for layer in range(self.num_layers):
            # weight_ih_l{layer}: (3*hidden_size, input_size)
            w_ih = param_dict[f'weight_ih_l{layer}']
            W_ih_list.append(w_ih)
            
            # weight_hh_l{layer}: (3*hidden_size, hidden_size)
            w_hh = param_dict[f'weight_hh_l{layer}']
            W_hh_list.append(w_hh)
            
            if self.bias:
                b_ih = param_dict[f'bias_ih_l{layer}']
                b_hh = param_dict[f'bias_hh_l{layer}']
            else:
                b_ih = torch.zeros(3 * self.hidden_size, device=x.device)
                b_hh = torch.zeros(3 * self.hidden_size, device=x.device)
            b_ih_list.append(b_ih)
            b_hh_list.append(b_hh)
        
        # Stack into tensors: (num_layers, 3*hidden_size, input_size) etc.
        W_ih = torch.stack(W_ih_list, dim=0)
        W_hh = torch.stack(W_hh_list, dim=0)
        b_ih = torch.stack(b_ih_list, dim=0)
        b_hh = torch.stack(b_hh_list, dim=0)
        
        # Call custom CUDA GRU
        h_n = self.gru_custom.gru_forward_cuda(
            x, h0, W_ih, W_hh, b_ih, b_hh,
            self.num_layers, seq_len, batch_size, self.input_size, self.hidden_size
        )
        
        return h_n