import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused linear + ReLU
fused_linear_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_linear_relu_kernel(const float* input, const float* weight, const float* bias, float* output,
                                         int batch_size, int in_features, int out_features) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < batch_size && col < out_features) {
        float sum = 0.0f;
        for (int k = 0; k < in_features; ++k) {
            sum += input[row * in_features + k] * weight[col * in_features + k];
        }
        sum += bias[col];
        output[row * out_features + col] = fmaxf(sum, 0.0f);
    }
}

torch::Tensor fused_linear_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::empty({batch_size, out_features}, input.options());
    
    dim3 block_size(16, 16);
    dim3 num_blocks((batch_size + block_size.x - 1) / block_size.x,
                    (out_features + block_size.y - 1) / block_size.y);
    
    fused_linear_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, in_features, out_features);
    
    return output;
}
"""

fused_linear_relu_cpp_source = "torch::Tensor fused_linear_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"

# Custom CUDA kernel for final linear layer (no activation)
linear_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void linear_kernel(const float* input, const float* weight, const float* bias, float* output,
                              int batch_size, int in_features, int out_features) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < batch_size && col < out_features) {
        float sum = 0.0f;
        for (int k = 0; k < in_features; ++k) {
            sum += input[row * in_features + k] * weight[col * in_features + k];
        }
        sum += bias[col];
        output[row * out_features + col] = sum;
    }
}

torch::Tensor linear_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::empty({batch_size, out_features}, input.options());
    
    dim3 block_size(16, 16);
    dim3 num_blocks((batch_size + block_size.x - 1) / block_size.x,
                    (out_features + block_size.y - 1) / block_size.y);
    
    linear_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, in_features, out_features);
    
    return output;
}
"""

linear_cpp_source = "torch::Tensor linear_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"

# Compile the custom CUDA kernels
fused_linear_relu = load_inline(
    name="fused_linear_relu",
    cpp_sources=fused_linear_relu_cpp_source,
    cuda_sources=fused_linear_relu_source,
    functions=["fused_linear_relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

linear = load_inline(
    name="linear",
    cpp_sources=linear_cpp_source,
    cuda_sources=linear_source,
    functions=["linear_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super(ModelNew, self).__init__()
        
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        
        # Create parameters for each layer
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        
        current_input_size = input_size
        for hidden_size in hidden_layer_sizes:
            self.weights.append(nn.Parameter(torch.empty(hidden_size, current_input_size)))
            self.biases.append(nn.Parameter(torch.empty(hidden_size)))
            current_input_size = hidden_size
        
        # Final layer
        self.weights.append(nn.Parameter(torch.empty(output_size, current_input_size)))
        self.biases.append(nn.Parameter(torch.empty(output_size)))
        
        # Initialize parameters
        self._init_parameters()
        
        # Store custom CUDA functions
        self.fused_linear_relu = fused_linear_relu
        self.linear = linear
    
    def _init_parameters(self):
        for weight in self.weights:
            nn.init.kaiming_uniform_(weight, a=0, mode='fan_in', nonlinearity='relu')
        for bias in self.biases:
            nn.init.uniform_(bias, -1.0 / (bias.size(0) ** 0.5), 1.0 / (bias.size(0) ** 0.5))
    
    def forward(self, x):
        # Apply hidden layers with fused linear + ReLU
        for i in range(len(self.hidden_layer_sizes)):
            x = self.fused_linear_relu.fused_linear_relu_cuda(x, self.weights[i], self.biases[i])
        
        # Apply final linear layer (no activation)
        x = self.linear.linear_cuda(x, self.weights[-1], self.biases[-1])
        
        return x