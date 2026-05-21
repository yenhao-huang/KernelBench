import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for LSTM cell forward pass
lstm_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void lstm_cell_kernel(
    const float* x, const float* h_prev, const float* c_prev,
    const float* W_ih, const float* W_hh, const float* b_ih, const float* b_hh,
    float* h_out, float* c_out,
    int batch_size, int input_size, int hidden_size
) {
    int b = blockIdx.x;
    int h = threadIdx.x;
    
    if (b < batch_size && h < hidden_size) {
        // Compute gates: i, f, g, o
        float i_gate = b_ih[h] + b_hh[h];
        float f_gate = b_ih[hidden_size + h] + b_hh[hidden_size + h];
        float g_gate = b_ih[2 * hidden_size + h] + b_hh[2 * hidden_size + h];
        float o_gate = b_ih[3 * hidden_size + h] + b_hh[3 * hidden_size + h];
        
        // Input-to-hidden contributions
        for (int k = 0; k < input_size; k++) {
            float x_val = x[b * input_size + k];
            i_gate += x_val * W_ih[h * input_size + k];
            f_gate += x_val * W_ih[(hidden_size + h) * input_size + k];
            g_gate += x_val * W_ih[(2 * hidden_size + h) * input_size + k];
            o_gate += x_val * W_ih[(3 * hidden_size + h) * input_size + k];
        }
        
        // Hidden-to-hidden contributions
        for (int k = 0; k < hidden_size; k++) {
            float h_val = h_prev[b * hidden_size + k];
            i_gate += h_val * W_hh[h * hidden_size + k];
            f_gate += h_val * W_hh[(hidden_size + h) * hidden_size + k];
            g_gate += h_val * W_hh[(2 * hidden_size + h) * hidden_size + k];
            o_gate += h_val * W_hh[(3 * hidden_size + h) * hidden_size + k];
        }
        
        // Apply activations
        i_gate = 1.0f / (1.0f + expf(-i_gate));  // sigmoid
        f_gate = 1.0f / (1.0f + expf(-f_gate));
        g_gate = tanhf(g_gate);
        o_gate = 1.0f / (1.0f + expf(-o_gate));
        
        // Update cell state and hidden state
        float c_new = f_gate * c_prev[b * hidden_size + h] + i_gate * g_gate;
        float h_new = o_gate * tanhf(c_new);
        
        c_out[b * hidden_size + h] = c_new;
        h_out[b * hidden_size + h] = h_new;
    }
}

torch::Tensor lstm_forward_cuda(
    torch::Tensor x, torch::Tensor h0, torch::Tensor c0,
    torch::Tensor W_ih, torch::Tensor W_hh, torch::Tensor b_ih, torch::Tensor b_hh,
    int num_layers
) {
    int batch_size = x.size(0);
    int seq_length = x.size(1);
    int input_size = x.size(2);
    int hidden_size = h0.size(2);
    
    auto h_out = torch::zeros({num_layers, batch_size, hidden_size}, x.options());
    auto c_out = torch::zeros({num_layers, batch_size, hidden_size}, x.options());
    
    // For each layer
    for (int layer = 0; layer < num_layers; layer++) {
        auto h_prev = h0[layer];
        auto c_prev = c0[layer];
        auto W_ih_layer = W_ih[layer];
        auto W_hh_layer = W_hh[layer];
        auto b_ih_layer = b_ih[layer];
        auto b_hh_layer = b_hh[layer];
        
        // For each time step
        for (int t = 0; t < seq_length; t++) {
            auto x_t = (layer == 0) ? x.select(1, t) : h_out[layer - 1].select(1, t);
            
            auto h_new = torch::zeros({batch_size, hidden_size}, x.options());
            auto c_new = torch::zeros({batch_size, hidden_size}, x.options());
            
            int threads = hidden_size;
            int blocks = batch_size;
            
            lstm_cell_kernel<<<blocks, threads>>>(
                x_t.data_ptr<float>(), h_prev.data_ptr<float>(), c_prev.data_ptr<float>(),
                W_ih_layer.data_ptr<float>(), W_hh_layer.data_ptr<float>(),
                b_ih_layer.data_ptr<float>(), b_hh_layer.data_ptr<float>(),
                h_new.data_ptr<float>(), c_new.data_ptr<float>(),
                batch_size, input_size, hidden_size
            );
            
            h_prev = h_new;
            c_prev = c_new;
        }
        
        h_out[layer] = h_prev;
        c_out[layer] = c_prev;
    }
    
    return h_out;  // Return final hidden states for all layers
}
"""

lstm_cpp_source = (
    "torch::Tensor lstm_forward_cuda("
    "torch::Tensor x, torch::Tensor h0, torch::Tensor c0,"
    "torch::Tensor W_ih, torch::Tensor W_hh, torch::Tensor b_ih, torch::Tensor b_hh,"
    "int num_layers"
    ");"
)

# Compile the inline CUDA code for LSTM
lstm_cuda = load_inline(
    name="lstm_cuda",
    cpp_sources=lstm_cpp_source,
    cuda_sources=lstm_cuda_source,
    functions=["lstm_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # LSTM weights and biases
        self.W_ih = nn.ParameterList()
        self.W_hh = nn.ParameterList()
        self.b_ih = nn.ParameterList()
        self.b_hh = nn.ParameterList()
        
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            self.W_ih.append(nn.Parameter(torch.randn(4 * hidden_size, layer_input_size)))
            self.W_hh.append(nn.Parameter(torch.randn(4 * hidden_size, hidden_size)))
            self.b_ih.append(nn.Parameter(torch.randn(4 * hidden_size)))
            self.b_hh.append(nn.Parameter(torch.randn(4 * hidden_size)))
        
        self.fc = nn.Linear(hidden_size, output_size)
        self.lstm_cuda = lstm_cuda
    
    def forward(self, x, h0, c0):
        # Stack weights into tensors for CUDA kernel
        W_ih_stacked = torch.stack([w for w in self.W_ih])
        W_hh_stacked = torch.stack([w for w in self.W_hh])
        b_ih_stacked = torch.stack([b for b in self.b_ih])
        b_hh_stacked = torch.stack([b for b in self.b_hh])
        
        # Run custom LSTM
        h_out = self.lstm_cuda.lstm_forward_cuda(
            x, h0, c0,
            W_ih_stacked, W_hh_stacked, b_ih_stacked, b_hh_stacked,
            self.num_layers
        )
        
        # Decode the hidden state of the last time step from the last layer
        out = self.fc(h_out[-1])  # h_out shape: (num_layers, batch_size, hidden_size)
        
        return h_out  # Return final hidden states for all layers (matching original: state[0])