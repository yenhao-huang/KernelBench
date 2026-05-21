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
    const float* __restrict__ x,          // input: [batch_size, input_size]
    const float* __restrict__ h_prev,     // previous hidden: [batch_size, hidden_size]
    const float* __restrict__ W_ih,       // input-hidden weights: [3*hidden_size, input_size]
    const float* __restrict__ W_hh,       // hidden-hidden weights: [3*hidden_size, hidden_size]
    const float* __restrict__ b_ih,       // input-hidden bias: [3*hidden_size]
    const float* __restrict__ b_hh,       // hidden-hidden bias: [3*hidden_size]
    float* __restrict__ h_new,            // new hidden: [batch_size, hidden_size]
    int batch_size,
    int input_size,
    int hidden_size
) {
    int b = blockIdx.x;  // batch index
    int h = threadIdx.x; // hidden dimension index
    
    if (b < batch_size && h < hidden_size) {
        // Compute gates: r, z, n
        // Each gate has hidden_size elements
        // Layout: W_ih is [3*hidden_size, input_size], W_hh is [3*hidden_size, hidden_size]
        // Gates: r (reset), z (update), n (new)
        
        float r_ih = 0.0f;
        float z_ih = 0.0f;
        float n_ih = 0.0f;
        
        // Compute input contributions for this hidden unit
        for (int i = 0; i < input_size; i++) {
            float xi = x[b * input_size + i];
            r_ih += W_ih[h * input_size + i] * xi;                    // r gate, input part
            z_ih += W_ih[(hidden_size + h) * input_size + i] * xi;    // z gate, input part
            n_ih += W_ih[(2 * hidden_size + h) * input_size + i] * xi; // n gate, input part
        }
        
        float r_hh = 0.0f;
        float z_hh = 0.0f;
        float n_hh = 0.0f;
        
        // Compute hidden contributions for this hidden unit
        for (int j = 0; j < hidden_size; j++) {
            float hj = h_prev[b * hidden_size + j];
            r_hh += W_hh[h * hidden_size + j] * hj;
            z_hh += W_hh[(hidden_size + h) * hidden_size + j] * hj;
            n_hh += W_hh[(2 * hidden_size + h) * hidden_size + j] * hj;
        }
        
        // Add biases
        float r = r_ih + r_hh + b_ih[h] + b_hh[h];
        float z = z_ih + z_hh + b_ih[hidden_size + h] + b_hh[hidden_size + h];
        float n = n_ih + n_hh + b_ih[2 * hidden_size + h] + b_hh[2 * hidden_size + h];
        
        // Apply sigmoid to r and z
        r = 1.0f / (1.0f + expf(-r));
        z = 1.0f / (1.0f + expf(-z));
        
        // Compute n with reset gate applied to hidden contribution
        // n = tanh(n_ih + r * n_hh + biases)
        // We already have n_ih and n_hh, but n_hh needs to be multiplied by r
        // Actually, the standard GRU formula: n = tanh(W_in * x + b_in + r * (W_hn * h + b_hn))
        // So we need to recompute n with r applied to hidden part
        // Let's recompute n properly:
        n = n_ih + b_ih[2 * hidden_size + h];  // input part + bias
        // Add r * (hidden part + bias)
        n += r * (n_hh + b_hh[2 * hidden_size + h]);
        n = tanhf(n);
        
        // New hidden state: h_new = (1 - z) * n + z * h_prev
        h_new[b * hidden_size + h] = (1.0f - z) * n + z * h_prev[b * hidden_size + h];
    }
}

// Multi-layer GRU forward pass
torch::Tensor gru_forward_cuda(
    torch::Tensor x,           // [seq_len, batch_size, input_size]
    torch::Tensor h0,          // [num_layers, batch_size, hidden_size]
    torch::Tensor W_ih,        // [num_layers, 3*hidden_size, input_size]
    torch::Tensor W_hh,        // [num_layers, 3*hidden_size, hidden_size]
    torch::Tensor b_ih,        // [num_layers, 3*hidden_size]
    torch::Tensor b_hh,        // [num_layers, 3*hidden_size]
    int num_layers,
    int seq_len,
    int batch_size,
    int input_size,
    int hidden_size
) {
    // Output tensor: [seq_len, batch_size, hidden_size]
    auto output = torch::zeros({seq_len, batch_size, hidden_size}, x.options());
    
    // Allocate intermediate hidden states for each layer
    auto h_cur = torch::zeros({num_layers, batch_size, hidden_size}, x.options());
    
    // Copy initial hidden state
    h_cur.copy_(h0);
    
    // For each time step
    for (int t = 0; t < seq_len; t++) {
        // Input for first layer is x[t]
        auto layer_input = x[t];  // [batch_size, input_size]
        
        for (int l = 0; l < num_layers; l++) {
            // Get weights and biases for this layer
            auto W_ih_l = W_ih[l];  // [3*hidden_size, input_size] or [3*hidden_size, hidden_size] for l>0
            auto W_hh_l = W_hh[l];  // [3*hidden_size, hidden_size]
            auto b_ih_l = b_ih[l];  // [3*hidden_size]
            auto b_hh_l = b_hh[l];  // [3*hidden_size]
            
            // Previous hidden for this layer
            auto h_prev_l = h_cur[l];  // [batch_size, hidden_size]
            
            // New hidden for this layer
            auto h_new_l = torch::zeros({batch_size, hidden_size}, x.options());
            
            int threads = hidden_size;
            int blocks = batch_size;
            
            gru_cell_kernel<<<blocks, threads>>>(
                layer_input.data_ptr<float>(),
                h_prev_l.data_ptr<float>(),
                W_ih_l.data_ptr<float>(),
                W_hh_l.data_ptr<float>(),
                b_ih_l.data_ptr<float>(),
                b_hh_l.data_ptr<float>(),
                h_new_l.data_ptr<float>(),
                batch_size,
                (l == 0) ? input_size : hidden_size,
                hidden_size
            );
            
            // Update current hidden for this layer
            h_cur[l] = h_new_l;
            
            // Output of this layer becomes input to next layer
            layer_input = h_new_l;
        }
        
        // Store output for this time step (last layer's hidden)
        output[t] = layer_input;
    }
    
    // Final hidden state is h_cur
    // Return both output and final hidden
    // We'll return a tuple, but since we can only return one tensor, we'll concatenate
    // Actually, let's return output and h_cur separately via a struct or just return output and have h_cur as an out parameter
    // For simplicity, we'll return a single tensor that contains both, but that's messy.
    // Better: return a vector of tensors.
    // But the load_inline expects a single tensor return type. We'll need to handle this differently.
    // Let's modify to return a tuple using pybind11.
    // Actually, we can return a std::vector<torch::Tensor> and it will be converted to a tuple in Python.
    return {output, h_cur};
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

# Compile the inline CUDA code
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
        
        # Create GRU parameters manually
        # Weights for each layer
        self.W_ih = nn.ParameterList()
        self.W_hh = nn.ParameterList()
        self.b_ih = nn.ParameterList()
        self.b_hh = nn.ParameterList()
        
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            # W_ih: [3*hidden_size, layer_input_size]
            self.W_ih.append(nn.Parameter(torch.randn(3 * hidden_size, layer_input_size)))
            # W_hh: [3*hidden_size, hidden_size]
            self.W_hh.append(nn.Parameter(torch.randn(3 * hidden_size, hidden_size)))
            if bias:
                self.b_ih.append(nn.Parameter(torch.randn(3 * hidden_size)))
                self.b_hh.append(nn.Parameter(torch.randn(3 * hidden_size)))
            else:
                self.b_ih.append(nn.Parameter(torch.zeros(3 * hidden_size), requires_grad=False))
                self.b_hh.append(nn.Parameter(torch.zeros(3 * hidden_size), requires_grad=False))
        
        self.gru_custom = gru_custom
    
    def forward(self, x, h0):
        # x: [seq_len, batch_size, input_size] if batch_first=False
        # h0: [num_layers, batch_size, hidden_size]
        
        if self.batch_first:
            x = x.transpose(0, 1)  # [batch_size, seq_len, input_size] -> [seq_len, batch_size, input_size]
        
        seq_len, batch_size, _ = x.shape
        
        # Stack weights into tensors
        W_ih_stacked = torch.stack([w for w in self.W_ih], dim=0)  # [num_layers, 3*hidden_size, input_size or hidden_size]
        W_hh_stacked = torch.stack([w for w in self.W_hh], dim=0)  # [num_layers, 3*hidden_size, hidden_size]
        b_ih_stacked = torch.stack([b for b in self.b_ih], dim=0)  # [num_layers, 3*hidden_size]
        b_hh_stacked = torch.stack([b for b in self.b_hh], dim=0)  # [num_layers, 3*hidden_size]
        
        # Call custom CUDA GRU
        result = self.gru_custom.gru_forward_cuda(
            x, h0,
            W_ih_stacked, W_hh_stacked,
            b_ih_stacked, b_hh_stacked,
            self.num_layers, seq_len, batch_size,
            self.input_size, self.hidden_size
        )
        
        # result is a tuple (output, h_n)
        output, h_n = result
        
        if self.batch_first:
            output = output.transpose(0, 1)  # [seq_len, batch_size, hidden_size] -> [batch_size, seq_len, hidden_size]
        
        return output