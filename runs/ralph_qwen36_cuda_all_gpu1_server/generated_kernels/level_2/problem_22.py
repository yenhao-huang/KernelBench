import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation:
# 1. Linear (Matmul + Bias)
# 2. Scale
# 3. Residual Add (x + x is effectively x * 2, but we keep it general as add(x, x))
# 4. Clamp
# 5. LogSumExp over dim 1
# 6. Mish Activation and Multiply

# Note: Implementing a fully fused kernel for all these steps including LogSumExp 
# is complex due to the reduction nature of LogSumExp. 
# However, we can fuse the Linear, Scale, Add, Clamp into one kernel, 
# and then perform LogSumExp and Mish in subsequent optimized kernels or even standard PyTorch ops if they are already fast.
# But to demonstrate "custom CUDA operators", let's create a highly optimized kernel for the heavy lifting:
# The Linear layer is the most expensive part (O(N^2)). We will replace it with a custom GEMM-like kernel 
# or simply use the fact that PyTorch's linear is already very optimized. 
# However, the prompt asks to replace operators to get speedups. 
# A common optimization is fusing the post-linear operations: Scale -> Add -> Clamp.
# Let's fuse: x = clamp(x * scale + x, min, max) into a single kernel after the matmul.
# And we can also optimize the Mish calculation if needed, but standard mish is reasonably fast.
# The biggest win here is likely fusing the element-wise ops after the linear layer to reduce memory bandwidth pressure.

# Let's define a fused kernel for: out = clamp(input * scale + input, min, max)
# This replaces: x * scale_factor; x + x; torch.clamp(x, ...)

fused_post_linear_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void fused_scale_add_clamp_kernel(
    const float* input, 
    float* output, 
    int batch_size, 
    int hidden_size, 
    float scale_factor, 
    float clamp_min, 
    float clamp_max) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * hidden_size;

    if (idx < total_elements) {
        float val = input[idx];
        // Scale and Add: x * scale + x = x * (scale + 1)
        // Optimization: precompute factor = scale_factor + 1.0f
        val = val * (scale_factor + 1.0f);
        
        // Clamp
        if (val < clamp_min) {
            val = clamp_min;
        } else if (val > clamp_max) {
            val = clamp_max;
        }
        
        output[idx] = val;
    }
}

torch::Tensor fused_scale_add_clamp_cuda(
    torch::Tensor input, 
    float scale_factor, 
    float clamp_min, 
    float clamp_max) 
{
    auto batch_size = input.size(0);
    auto hidden_size = input.size(1);
    auto output = torch::empty_like(input);

    const int block_size = 256;
    int total_elements = batch_size * hidden_size;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_scale_add_clamp_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        batch_size, 
        hidden_size, 
        scale_factor, 
        clamp_min, 
        clamp_max
    );

    return output;
}
"""

fused_post_linear_cpp_source = (
    "torch::Tensor fused_scale_add_clamp_cuda(torch::Tensor input, float scale_factor, float clamp_min, float clamp_max);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_post_linear",
    cpp_sources=fused_post_linear_cpp_source,
    cuda_sources=fused_post_linear_source,
    functions=["fused_scale_add_clamp_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA operators for the post-linear element-wise operations.
    Fuses: Scale, Add (Residual), and Clamp into a single kernel to reduce memory traffic.
    """
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        # Keep the linear layer as is, or replace with custom if desired. 
        # PyTorch's Linear is already highly optimized (cuBLAS). 
        # The bottleneck here is likely memory bandwidth for the element-wise ops on large tensors.
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, hidden_size).
        """
        # 1. Matrix Multiplication + Bias
        x = self.matmul(x)
        
        # 2. Fused Scale, Add, and Clamp using custom CUDA kernel
        # This replaces: x * scale_factor; x + x; torch.clamp(x, ...)
        x = fused_ops.fused_scale_add_clamp_cuda(x, self.scale_factor, self.clamp_min, self.clamp_max)
        
        # 3. LogSumExp over dim 1
        # We keep this as standard PyTorch op as it involves a reduction which is complex to fuse 
        # with the previous element-wise ops without significant complexity overhead.
        x = torch.logsumexp(x, dim=1, keepdim=True)
        
        # 4. Mish Activation and Multiply
        # Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
        # We can implement a custom Mish kernel for further optimization if needed, 
        # but standard F.mish is reasonably fast. Let's stick to standard for simplicity 
        # unless we want to fuse it with the multiplication.
        # However, to show "custom operators", let's create a fused Mish multiply kernel.
        
        x = self._mish_multiply(x)
        
        return x

    def _mish_multiply(self, x):
        """
        Custom CUDA kernel for: out = x * mish(x)
        mish(x) = x * tanh(softplus(x))
        """
        # We need to define another inline module or use the existing one if we extend it.
        # Since load_inline creates a new module, let's define the Mish kernel separately 
        # and load it into a separate module or extend the previous one.
        # For clarity and modularity, let's define a second inline load for Mish.
        
        mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float softplus(float x) {
    return (x > 20.0f) ? x : log1pf(expf(x));
}

__global__ void mish_multiply_kernel(
    const float* input, 
    float* output, 
    int size) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        // Mish(x) = x * tanh(softplus(x))
        float sp = softplus(x);
        float tanh_sp = tanhf(sp);
        float mish_x = x * tanh_sp;
        
        // Final output: x * mish(x)
        output[idx] = x * mish_x;
    }
}

torch::Tensor mish_multiply_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    mish_multiply_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""
        mish_cpp_source = "torch::Tensor mish_multiply_cuda(torch::Tensor input);"
        
        # Load the Mish module
        mish_module = load_inline(
            name="mish_op",
            cpp_sources=mish_cpp_source,
            cuda_sources=mish_source,
            functions=["mish_multiply_cuda"],
            verbose=False,
            extra_cflags=["-O3"],
            extra_ldflags=[""]
        )
        
        return mish_module.mish_multiply_cuda(x)

def get_inputs():
    # randomly generate input tensors based on the model architecture
    a = torch.rand(batch_size, input_size).cuda()
    return [a]

def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    return [input_size, hidden_size, scale_factor, clamp_min, clamp_max]

# Global variables to match the example structure's expectations if needed, 
# though the class definition is the primary output.
batch_size = 1024
input_size = 8192
hidden_size = 8192
scale_factor = 2.0
clamp_min = -10.0
clamp_max = 10.0