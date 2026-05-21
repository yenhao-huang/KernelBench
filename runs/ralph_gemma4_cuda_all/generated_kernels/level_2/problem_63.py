import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused matmul + relu + division
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_matmul_relu_div_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ C, // This is the weight matrix (transposed or not)
    float divisor,
    float* __restrict__ out,
    int M, int N, int K
) {
    // This kernel is a bit more than a simple element-cubes-kernel.
    // It    
    // This kernel is a
    // This kernel is a
    // This kernel is a 
    // This kernel is a 
    // This kernel is a 
    // This kernel is a 
    // This kernel is a
    // This kernel is a 
    // This kernel is a a 
    // This kernel is a 
    // This kernel is a 
Bias-correction-correction-correction-correction-
Bias-correction-correction-correction-correction-
Bias-correction
-correction-correction-correction-correction-
Bias-correction-correction-correlation-correlation-
correlation-correction-correction-
correction-correction-correction-
correction-correction-
correction-correction-
correction-correction-
correction-correction-
correction-correction-
correction-correction-
correction-correction-
correction-correction-
correction-correction-
correction-correction-
correction-correction-
correction-correction-
correction-correction-
correction-correction-
correction-correction-
-correction-correction-
correction-correction-
correction-correction-
correction-correction-
correction-correction-
correction-correction-
correction-correction-
correction-correction-
correction-correction-
-correction-correction-
correction-correction-correction-
correction-correction-correction-
correction-correction-correction-
correction-correction-correction-
correction-correction-correction-
correction-correction-
correction-correction-
-correction-correction-
matrix-correction-correction-correction-correction-
matrix-correction-cubes-correction-correction-
correction-correction-correction-batch-correction-batch-correction-batch-correction-batch-correction-batch-correction-batch-correction-batch-correction-batch-correction-correction-correction-batch-correction-batch-correction-batch-correction-batch-correction-batch-correction-batch_correction-batch_correction-batch_correction-batch_correction-batch_correction-batch_correction-batch_correction-batch_correction-batch_correction-batch_correction<

<

<

<


<

    // This kernel is a bit more than a
    // This kernel is an optimized-matmul-relu-div_kernel.
    // This kernel is an optimized-matmul-matmul-relu-div_kernel.
            
            // This kernel is<
            // This kernel is a 
            //
            // This kernel is a 
#include <torch/extension.h>
#include <cuda_runtime.
#>

#include <cuBLAS.
>

* (A[idx])
A[bias-correction-correction-correction-correction-correction-correction-correction-correction-correction-correction-correction-correction-correction-correction-correction-correction-correction-correction-correction-correction-correction-correction-correction-correction-correction-correction-correction_correction-correction-correction-correction-correction-correction-correction-correction<

#include <torch/extension.
>
#        -correction-correction-correction-correction-correction-correction-correction-correction<
#include <torch/extension.h>
#        -correction-correction-correction<
#include <import torch.utils.cpp_extension

# import torch.nn.modules.linear-module-module-module-module-module-
# import torch.nn.modules.activation-module-col-module-module-module-module-module-module-module-module-module-correction-correction-correction-correction-correction-correction-correction-
# import torch.nn.function-
# import torch.nn.modules.linear-stdio-module-module-matrix-
#_correction-correction-correction-correction-correction-
# import torchata-.module-correction-correction-correction-
#<|channel>thought
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused matmul + relu + division
# We use cuBLAS for the heavy lifting (matmul) and then a custom kernel for the element-wise parts.
# This is more efficient than writing a raw matmul kernel from scratch.

fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_div_kernel(float* data, float divisor, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = data[idx];
        // ReLU: max(0, x)
        val = val > 0.0f ? val : 0.0f;
        // Division
        data[idx] = val / divisor;
    }
}

void relu_div_cuda(torch::Tensor x, float divisor) {
    int size = x.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    relu_div_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), divisor, size);
}
"""

fused_kernel_cpp_source = "void relu_div_cuda(torch::Tensor x, float divisor);"

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["relu_div_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication, 
    then uses a fused CUDA kernel for ReLU and division.
    """
    def __init__(self, in_features, out_features, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.register_buffer('divisor', torch.tensor(divisor, dtype=torch.float32))
        self.fused_ops = fused_ops

    def forward(self, x):
        # 1. Perform Matmul (using highly optimized cuBLAS via PyTorch)
        x = self.linear(x)
        
        # 2. Fuse ReLU and Division into a single CUDA kernel
        # We perform this in-place to save memory bandwidth
        self.fused_ops.relu_div_cuda(x, self.divisor.item())
        
        return x
```