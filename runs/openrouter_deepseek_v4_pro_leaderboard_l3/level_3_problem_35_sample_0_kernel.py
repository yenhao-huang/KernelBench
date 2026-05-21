import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for LSTM cell forward pass (single layer, single time step)
lstm_cell_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void lstm_cell_kernel(
    const float* __restrict__ x,        // input: (batch_size, input_size)
    const float* __restrict__ h_prev,   // previous hidden: (batch_size, hidden_size)
    const float* __restrict__ c_prev,   // previous cell: (batch_size, hidden_size)
    const float* __restrict__ W_ih,     // input-hidden weights: (4*hidden_size, input_size)
    const float* __restrict__ W_hh,     // hidden-hidden weights: (4*hidden_size, hidden_size)
    const float* __restrict__ b_ih,     // input-hidden bias: (4*hidden_size)
    const float* __restrict__ b_hh,     // hidden-hidden bias: (4*hidden_size)
    float* __restrict__ h_out,          // output hidden: (batch_size, hidden_size)
    float* __restrict__ c_out,          // output cell: (batch_size, hidden_size)
    int batch_size,
    int input_size,
    int hidden_size
) {
    int b = blockIdx.x;  // batch index
    int h = threadIdx.x; // hidden dimension index

    if (b >= batch_size || h >= hidden_size) return;

    // Compute gates: i, f, g, o
    // Each gate is computed as: gate = sum_j W_ih[gate_offset + h][j] * x[b][j] + sum_j W_hh[gate_offset + h][j] * h_prev[b][j] + b_ih[gate_offset + h] + b_hh[gate_offset + h]
    // We'll compute all four gates for this hidden unit

    float i_gate = 0.0f, f_gate = 0.0f, g_gate = 0.0f, o_gate = 0.0f;

    // Input-hidden contributions
    for (int j = 0; j < input_size; ++j) {
        float x_val = x[b * input_size + j];
        i_gate += W_ih[h * input_size + j] * x_val;
        f_gate += W_ih[(hidden_size + h) * input_size + j] * x_val;
        g_gate += W_ih[(2 * hidden_size + h) * input_size + j] * x_val;
        o_gate += W_ih[(3 * hidden_size + h) * input_size + j] * x_val;
    }

    // Hidden-hidden contributions
    for (int j = 0; j < hidden_size; ++j) {
        float h_val = h_prev[b * hidden_size + j];
        i_gate += W_hh[h * hidden_size + j] * h_val;
        f_gate += W_hh[(hidden_size + h) * hidden_size + j] * h_val;
        g_gate += W_hh[(2 * hidden_size + h) * hidden_size + j] * h_val;
        o_gate += W_hh[(3 * hidden_size + h) * hidden_size + j] * h_val;
    }

    // Add biases
    i_gate += b_ih[h] + b_hh[h];
    f_gate += b_ih[hidden_size + h] + b_hh[hidden_size + h];
    g_gate += b_ih[2 * hidden_size + h] + b_hh[2 * hidden_size + h];
    o_gate += b_ih[3 * hidden_size + h] + b_hh[3 * hidden_size + h];

    // Apply activations
    i_gate = 1.0f / (1.0f + expf(-i_gate));  // sigmoid
    f_gate = 1.0f / (1.0f + expf(-f_gate));
    g_gate = tanhf(g_gate);
    o_gate = 1.0f / (1.0f + expf(-o_gate));

    // Update cell and hidden
    float c_val = f_gate * c_prev[b * hidden_size + h] + i_gate * g_gate;
    float h_val = o_gate * tanhf(c_val);

    c_out[b * hidden_size + h] = c_val;
    h_out[b * hidden_size + h] = h_val;
}

torch::Tensor lstm_cell_cuda(
    torch::Tensor x,
    torch::Tensor h_prev,
    torch::Tensor c_prev,
    torch::Tensor W_ih,
    torch::Tensor W_hh,
    torch::Tensor b_ih,
    torch::Tensor b_hh
) {
    int batch_size = x.size(0);
    int input_size = x.size(1);
    int hidden_size = h_prev.size(1);

    auto h_out = torch::empty_like(h_prev);
    auto c_out = torch::empty_like(c_prev);

    dim3 blocks(batch_size);
    dim3 threads(hidden_size);

    lstm_cell_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        h_prev.data_ptr<float>(),
        c_prev.data_ptr<float>(),
        W_ih.data_ptr<float>(),
        W_hh.data_ptr<float>(),
        b_ih.data_ptr<float>(),
        b_hh.data_ptr<float>(),
        h_out.data_ptr<float>(),
        c_out.data_ptr<float>(),
        batch_size,
        input_size,
        hidden_size
    );

    return std::make_tuple(h_out, c_out);
}
"""

lstm_cell_cpp_source = """
torch::Tensor lstm_cell_cuda(
    torch::Tensor x,
    torch::Tensor h_prev,
    torch::Tensor c_prev,
    torch::Tensor W_ih,
    torch::Tensor W_hh,
    torch::Tensor b_ih,
    torch::Tensor b_hh
);
"""

# Compile the custom LSTM cell
lstm_cell_module = load_inline(
    name="lstm_cell",
    cpp_sources=lstm_cell_cpp_source,
    cuda_sources=lstm_cell_source,
    functions=["lstm_cell_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Custom CUDA kernel for linear layer (matrix multiply + bias)
linear_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void linear_kernel(
    const float* __restrict__ input,   // (batch_size, in_features)
    const float* __restrict__ weight,  // (out_features, in_features)
    const float* __restrict__ bias,    // (out_features)
    float* __restrict__ output,        // (batch_size, out_features)
    int batch_size,
    int in_features,
    int out_features
) {
    int b = blockIdx.x;   // batch index
    int o = threadIdx.x;  // output feature index

    if (b >= batch_size || o >= out_features) return;

    float sum = bias[o];
    for (int i = 0; i < in_features; ++i) {
        sum += input[b * in_features + i] * weight[o * in_features + i];
    }
    output[b * out_features + o] = sum;
}

torch::Tensor linear_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, input.options());

    dim3 blocks(batch_size);
    dim3 threads(out_features);

    linear_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );

    return output;
}
"""

linear_cpp_source = """
torch::Tensor linear_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
);
"""

# Compile the custom linear layer
linear_module = load_inline(
    name="linear_cuda",
    cpp_sources=linear_cpp_source,
    cuda_sources=linear_source,
    functions=["linear_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class CustomLSTMLayer(nn.Module):
    """Single LSTM layer using custom CUDA kernel, processing all time steps sequentially."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # LSTM parameters: W_ih (4*hidden_size, input_size), W_hh (4*hidden_size, hidden_size)
        # biases: b_ih (4*hidden_size), b_hh (4*hidden_size)
        self.W_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.b_ih = nn.Parameter(torch.randn(4 * hidden_size))
        self.b_hh = nn.Parameter(torch.randn(4 * hidden_size))
        self._lstm_cell = lstm_cell_module

    def forward(self, x, h0, c0):
        # x: (batch_size, seq_len, input_size)
        # h0, c0: (batch_size, hidden_size) - we'll handle single layer, so squeeze num_layers dim
        batch_size, seq_len, _ = x.shape
        h = h0
        c = c0
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch_size, input_size)
            h, c = self._lstm_cell.lstm_cell_cuda(x_t, h, c, self.W_ih, self.W_hh, self.b_ih, self.b_hh)
            outputs.append(h.unsqueeze(1))  # (batch_size, 1, hidden_size)
        out = torch.cat(outputs, dim=1)  # (batch_size, seq_len, hidden_size)
        return out, (h, c)


class CustomLSTM(nn.Module):
    """Multi-layer LSTM using custom CUDA kernels, with optional dropout."""
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.layers.append(CustomLSTMLayer(layer_input_size, hidden_size))

    def forward(self, x, h0, c0):
        # x: (batch_size, seq_len, input_size)
        # h0, c0: (num_layers, batch_size, hidden_size)
        batch_size = x.size(0)
        h_out = []
        c_out = []
        for i, layer in enumerate(self.layers):
            h_i = h0[i]  # (batch_size, hidden_size)
            c_i = c0[i]
            x, (h_i, c_i) = layer(x, h_i, c_i)
            if self.dropout is not None and i < self.num_layers - 1:
                x = self.dropout(x)
            h_out.append(h_i)
            c_out.append(c_i)
        h_out = torch.stack(h_out, dim=0)  # (num_layers, batch_size, hidden_size)
        c_out = torch.stack(c_out, dim=0)
        return x, (h_out, c_out)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(ModelNew, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = CustomLSTM(input_size, hidden_size, num_layers, dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self._linear_cuda = linear_module

    def forward(self, x, h0=None, c0=None):
        batch_size = x.size(0)

        if h0 is None:
            h0 = torch.randn(self.num_layers, batch_size, self.hidden_size, device=x.device)
        if c0 is None:
            c0 = torch.randn(self.num_layers, batch_size, self.hidden_size, device=x.device)

        out, _ = self.lstm(x, h0, c0)  # out: (batch_size, seq_length, hidden_size)
        # Take last time step
        out = out[:, -1, :]  # (batch_size, hidden_size)
        # Use custom linear layer
        out = self._linear_cuda.linear_cuda(out, self.fc.weight, self.fc.bias)
        return out