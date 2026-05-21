import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for fused last-timestep extraction + linear transformation
fused_last_linear_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_last_linear_kernel(
    const float* lstm_out,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int output_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * output_size;
    if (idx >= total) return;
    
    int batch = idx / output_size;
    int out_idx = idx % output_size;
    
    const float* input_ptr = lstm_out + batch * seq_len * hidden_dim + (seq_len - 1) * hidden_dim;
    float sum = bias[out_idx];
    const float* weight_row = weight + out_idx * hidden_dim;
    for (int i = 0; i < hidden_dim; i++) {
        sum += weight_row[i] * input_ptr[i];
    }
    output[idx] = sum;
}

torch::Tensor fused_last_linear_cuda(
    torch::Tensor lstm_out,
    torch::Tensor weight,
    torch::Tensor bias,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int output_size)
{
    auto output = torch::empty({batch_size, output_size}, lstm_out.options());
    
    const int block_size = 256;
    const int total = batch_size * output_size;
    const int num_blocks = (total + block_size - 1) / block_size;
    
    fused_last_linear_kernel<<<num_blocks, block_size>>>(
        lstm_out.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        seq_len,
        hidden_dim,
        output_size
    );
    
    return output;
}
"""

fused_last_linear_cpp_source = "torch::Tensor fused_last_linear_cuda(torch::Tensor lstm_out, torch::Tensor weight, torch::Tensor bias, int batch_size, int seq_len, int hidden_dim, int output_size);"

# Compile the inline CUDA code
fused_last_linear = load_inline(
    name="fused_last_linear",
    cpp_sources=fused_last_linear_cpp_source,
    cuda_sources=fused_last_linear_source,
    functions=["fused_last_linear_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(ModelNew, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.fused_last_linear = fused_last_linear

    def forward(self, x, h0, c0):
        # Forward propagate LSTM
        out, hn = self.lstm(x, (h0, c0))  # out: (batch, seq_len, hidden_size*2)
        
        # Ensure contiguity for custom kernel
        out = out.contiguous()
        
        batch_size = out.size(0)
        seq_len = out.size(1)
        hidden_dim = out.size(2)  # hidden_size * 2
        output_size = self.fc.out_features
        
        # Fused last-timestep extraction + linear transformation
        out = self.fused_last_linear.fused_last_linear_cuda(
            out, self.fc.weight, self.fc.bias,
            batch_size, seq_len, hidden_dim, output_size
        )
        return out