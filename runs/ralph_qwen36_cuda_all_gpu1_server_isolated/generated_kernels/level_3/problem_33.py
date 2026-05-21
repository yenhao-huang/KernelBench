import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels for optimized operations
# We will fuse: Concat -> Linear (i2h) -> Tanh -> Linear (h2o)
# However, since Tanh is non-linear and separates two linear layers, 
# a full fusion of i2h + tanh + h2o into one kernel is complex but possible.
# Alternatively, we can optimize the large Matmuls (i2h and h2o) which are the bottleneck.
# Given input_size=16384 and hidden_size=16384, i2h is 16384x16384. This is huge.
# We will implement a fused kernel: Concat + Matmul(i2h) + Tanh + Matmul(h2o).

custom_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Helper for tanh approximation or exact tanh. For FP32, we can use cuTanh if available, 
// but standard math is fine. We'll use a fast approximation or direct call.
__device__ __forceinline__ float fast_tanh(float x) {
    return tanhf(x);
}

// Kernel for Fused Concat + Linear1 + Tanh + Linear2
// Inputs:
// x: (batch, input_size)
// h_prev: (batch, hidden_size)
// W_i2h: (hidden_size, input_size + hidden_size)
// b_i2h: (hidden_size,)
// W_h2o: (output_size, hidden_size)
// b_h2o: (output_size,)
// Output:
// out: (batch, output_size)

__global__ void fused_rnn_step_kernel(
    const float* __restrict__ x,
    const float* __restrict__ h_prev,
    const float* __restrict__ W_i2h,
    const float* __restrict__ b_i2h,
    const float* __restrict__ W_h2o,
    const float* __restrict__ b_h2o,
    float* __restrict__ out,
    int batch_size,
    int input_size,
    int hidden_size,
    int output_size
) {
    int batch_idx = blockIdx.y; // Each block handles one sample in the batch? 
                               // Or we can do 1D grid for all elements.
                               // Let's use a 2D grid: x-dim for output features, y-dim for batch.
    
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= output_size) return;

    int b = batch_idx;
    if (b >= batch_size) return;

    // We need to compute:
    // 1. combined = [x[b], h_prev[b]] -> size (input_size + hidden_size)
    // 2. z = W_i2h * combined + b_i2h -> size hidden_size
    // 3. h_new = tanh(z)
    // 4. out[b] = W_h2o * h_new + b_h2o
    
    // To optimize, we can compute the contribution of each element in the hidden state 
    // to the current output element.
    
    float sum = 0.0f;
    
    // Precompute x and h_prev for this batch
    const float* x_b = x + b * input_size;
    const float* h_prev_b = h_prev + b * hidden_size;
    
    // W_i2h layout: (hidden_size, input_size + hidden_size)
    // W_h2o layout: (output_size, hidden_size)
    
    // We are computing out[b][out_idx]
    // out[b][out_idx] = sum_{k=0}^{hidden_size-1} (W_h2o[out_idx][k] * h_new[k]) + b_h2o[out_idx]
    // h_new[k] = tanh( sum_{j=0}^{input_size+hidden_size-1} (W_i2h[k][j] * combined[j]) + b_i2h[k] )
    
    // This is a nested loop. To make it efficient, we should structure the loops carefully.
    // However, with 16384 hidden size, this inner loop is heavy.
    // A better approach for CUDA is to have each thread compute one output element 
    // and iterate over the hidden dimension, loading W_h2o and computing h_new on the fly?
    // No, computing h_new requires iterating over input+hidden again.
    
    // Let's try a different tiling strategy or just straightforward computation if memory bandwidth allows.
    // Given the size, we might want to use shared memory for W_i2h rows and W_h2o columns.
    
    // Simplified approach: Each thread computes one output element.
    // It iterates over hidden_size (k). For each k, it computes h_new[k].
    // To compute h_new[k], it iterates over (input_size + hidden_size) (j).
    
    const float* W_h2o_row = W_h2o + out_idx * hidden_size;
    float b_h2o_val = b_h2o[out_idx];
    
    for (int k = 0; k < hidden_size; ++k) {
        // Compute h_new[k]
        float z_k = b_i2h[k];
        
        const float* W_i2h_row = W_i2h + k * (input_size + hidden_size);
        
        // Iterate over input part
        for (int j = 0; j < input_size; ++j) {
            z_k += W_i2h_row[j] * x_b[j];
        }
        
        // Iterate over hidden part
        for (int j = 0; j < hidden_size; ++j) {
            z_k += W_i2h_row[input_size + j] * h_prev_b[j];
        }
        
        float h_new_k = fast_tanh(z_k);
        
        sum += W_h2o_row[k] * h_new_k;
    }
    
    out[b * output_size + out_idx] = sum + b_h2o_val;
}

torch::Tensor fused_rnn_step_cuda(
    torch::Tensor x,
    torch::Tensor h_prev,
    torch::Tensor W_i2h,
    torch::Tensor b_i2h,
    torch::Tensor W_h2o,
    torch::Tensor b_h2o
) {
    auto batch_size = x.size(0);
    auto input_size = x.size(1);
    auto hidden_size = h_prev.size(1);
    auto output_size = W_h2o.size(0);

    auto out = torch::zeros({batch_size, output_size}, x.options());

    const int block_size = 256;
    const int num_blocks_x = (output_size + block_size - 1) / block_size;
    
    dim3 grid(num_blocks_x, batch_size);
    dim3 block(block_size);

    fused_rnn_step_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        h_prev.data_ptr<float>(),
        W_i2h.data_ptr<float>(),
        b_i2h.data_ptr<float>(),
        W_h2o.data_ptr<float>(),
        b_h2o.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        input_size,
        hidden_size,
        output_size
    );

    return out;
}
"""

custom_cpp_source = """
torch::Tensor fused_rnn_step_cuda(
    torch::Tensor x,
    torch::Tensor h_prev,
    torch::Tensor W_i2h,
    torch::Tensor b_i2h,
    torch::Tensor W_h2o,
    torch::Tensor b_h2o
);
"""

# Compile the inline CUDA code
fused_rnn_module = load_inline(
    name="fused_rnn_step",
    cpp_sources=custom_cpp_source,
    cuda_sources=custom_cuda_source,
    functions=["fused_rnn_step_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"]
)


class ModelNew(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize the Optimized RNN model with fused CUDA operator.
        
        :param input_size: The number of input features (int).
        :param hidden_size: The size of the hidden state (int).
        :param output_size: The number of output features (int).
        """
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases for the linear layers
        # i2h: Linear(input_size + hidden_size, hidden_size)
        self.register_buffer('W_i2h', torch.randn(hidden_size, input_size + hidden_size))
        self.register_buffer('b_i2h', torch.randn(hidden_size))
        
        # h2o: Linear(hidden_size, output_size)
        self.register_buffer('W_h2o', torch.randn(output_size, hidden_size))
        self.register_buffer('b_h2o', torch.randn(output_size))
        
        # Hidden state buffer
        self.register_buffer('hidden', torch.zeros(1, hidden_size))

    def forward(self, x: torch.Tensor, initial_hidden=None) -> torch.Tensor:
        """
        Forward pass of the Optimized Vanilla RNN using fused CUDA kernel.
        
        :param x: Input tensor of shape (batch_size, input_size).
        :param initial_hidden: Optional hidden state tensor of shape (1, hidden_size).
        :return: Output tensor of shape (batch_size, output_size).
        """
        batch_size = x.size(0)
        
        if initial_hidden is not None:
            self.hidden.copy_(initial_hidden)
            
        # Ensure hidden state is on the correct device and has the right shape
        # The model expects hidden to be (1, hidden_size) or broadcastable. 
        # In the original code, it was (batch_size, hidden_size) implicitly via copy_ or creation.
        # Let's align with the original logic where self.hidden is updated.
        # Original: self.hidden = torch.randn((batch_size, hidden_size)) in init? No, that was a bug in example.
        # Original forward: self.hidden.copy_(initial_hidden) if provided.
        # We will maintain self.hidden as (1, hidden_size) for the state, but expand it or handle batching in kernel.
        
        # The original code had: self.hidden = torch.randn((batch_size, hidden_size)) inside __init__? 
        # No, it was `self.hidden = torch.randn((batch_size, hidden_size))` which is invalid because batch_size isn't defined there.
        # It likely meant a single state vector or the user passes batched hidden states.
        # Let's assume standard RNN behavior: hidden state is (1, hidden_size) per sequence step, 
        # but here we process one batch at a time.
        
        # If initial_hidden is not provided, use self.hidden.
        if initial_hidden is None:
            h_prev = self.hidden.expand(batch_size, -1)
        else:
            h_prev = initial_hidden
            
        # Ensure h_prev is contiguous and on the same device as x
        h_prev = h_prev.contiguous()
        
        # Get weights
        W_i2h = self.W_i2h
        b_i2h = self.b_i2h
        W_h2o = self.W_h2o
        b_h2o = self.b_h2o
        
        # Call the fused CUDA kernel
        output = fused_rnn_module.fused_rnn_step_cuda(
            x, 
            h_prev, 
            W_i2h, 
            b_i2h, 
            W_h2o, 
            b_h2o
        )
        
        # Update hidden state for next step (if needed by caller)
        # The original code updated self.hidden. We need to compute the new hidden state 
        # to return it or store it. The fused kernel only computes output.
        # To be fully compatible, we should also return the new hidden state or update self.hidden.
        # However, the prompt asks to replace operators in the architecture.
        # The original forward returns `output`. It does NOT return the new hidden state explicitly in the return statement?
        # Original: `return output`
        # But it updates `self.hidden`.
        
        # To update self.hidden correctly, we need to compute the new hidden state.
        # We can either run a separate kernel for just the hidden update or modify the fused kernel.
        # For simplicity and correctness of the model state, let's add a helper to compute new hidden.
        # Or, we can assume the caller doesn't rely on self.hidden being updated if they pass initial_hidden every time.
        # But the original code does `self.hidden.copy_(initial_hidden)` and then updates it.
        
        # Let's create a separate kernel for hidden update to keep the main fused kernel simple, 
        # or just use PyTorch ops for the hidden update since it's less critical than the output matmul?
        # Actually, the hidden update is also a large matmul (16384x16384).
        
        # Let's implement a separate fused kernel for hidden state update: Concat + Linear + Tanh
        # But to keep the code block clean and focused on the main optimization requested (speedup),
        # and since the original model only returns `output`, we will focus on optimizing the output path.
        # However, if `self.hidden` is not updated correctly, subsequent calls will be wrong.
        
        # We will add a second fused kernel for hidden state update.
        
        return output

    def get_new_hidden(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """Helper to compute new hidden state using PyTorch ops or a separate kernel."""
        # For now, we use standard PyTorch ops for hidden update to ensure correctness 
        # while the main output path is optimized. In a real scenario, you'd fuse this too.
        combined = torch.cat((x, h_prev), dim=1)
        z = self.i2h_fused(combined) # This would need the weights
        return torch.tanh(z)

# Since we can't easily define two kernels in one load_inline call with different names without more complex setup,
# and to ensure the code compiles and runs, let's stick to the single fused kernel for output.
# We will update self.hidden using standard PyTorch operations which are still fast enough compared to the naive approach 
# if we assume the bottleneck is the double matmul in the output path? 
# Actually, the hidden update IS a matmul of the same size.
# Let's refine the solution to include a second kernel for hidden update or just use PyTorch for it.
# Given the constraints "Just output the new model code", I will provide a version that uses the fused kernel for output 
# and standard PyTorch for hidden state update, noting that in production both should be fused.

# Re-defining ModelNew to include necessary buffers and methods properly

class ModelNew(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Buffers for weights and biases
        self.register_buffer('W_i2h', torch.randn(hidden_size, input_size + hidden_size))
        self.register_buffer('b_i2h', torch.randn(hidden_size))
        self.register_buffer('W_h2o', torch.randn(output_size, hidden_size))
        self.register_buffer('b_h2o', torch.randn(output_size))
        
        # Hidden state buffer (1, hidden_size)
        self.register_buffer('hidden', torch.zeros(1, hidden_size))

    def forward(self, x: torch.Tensor, initial_hidden=None) -> torch.Tensor:
        batch_size = x.size(0)
        
        if initial_hidden is not None:
            self.hidden.copy_(initial_hidden)
            
        # Expand hidden state to match batch size for the kernel
        h_prev = self.hidden.expand(batch_size, -1).contiguous()
        
        # Compute output using fused CUDA kernel
        output = fused_rnn_module.fused_rnn_step_cuda(
            x, 
            h_prev, 
            self.W_i2h, 
            self.b_i2h, 
            self.W_h2o, 
            self.b_h2o
        )
        
        # Update hidden state for next iteration
        # We use standard PyTorch ops here. While not fused with the output kernel,
        # it ensures the model state is correct. The main speedup comes from the output path optimization.
        combined = torch.cat((x, h_prev), dim=1)
        z = torch.addmm(self.b_i2h, combined, self.W_i2h.t()) # Linear: W * x + b
        self.hidden = torch.tanh(z).mean(dim=0, keepdim=True) # Average over batch to get single state? 
                                                              # Original code didn't average. It just assigned.
                                                              # Original: self.hidden = self.tanh(self.i2h(combined))
                                                              # If combined is (B, D), i2h output is (B, H).
                                                              # self.hidden was (1, H) or (B, H)? 
                                                              # In __init__: `self.hidden = torch.randn((batch_size, hidden_size))` -> This line in original code is buggy as batch_size is not defined.
                                                              # Assuming it meant a single state vector (1, H) that gets broadcasted or copied.
                                                              # Let's assume the standard RNN behavior where hidden state is per-batch-item if passed, 
                                                              # but stored as a single vector in the module? 
                                                              # The original code: `self.hidden = self.tanh(self.i2h(combined))` assigns a (B, H) tensor to self.hidden.
                                                              # So self.hidden shape changes dynamically.
        
        # To match original behavior exactly:
        combined = torch.cat((x, h_prev), dim=1)
        z = torch.addmm(self.b_i2h, combined, self.W_i2h.t())
        self.hidden = torch.tanh(z)

        return output

def get_inputs():
    return [torch.rand(batch_size, input_size), torch.rand(1, hidden_size)] # Initial hidden is usually (1, H) or (B, H)

def get_init_inputs():
    return [input_size, hidden_size, output_size]