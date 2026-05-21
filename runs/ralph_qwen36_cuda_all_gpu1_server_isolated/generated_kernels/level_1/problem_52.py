import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for argmin along a specific dimension
argmin_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

// Kernel to find the index of the minimum value along the last dimension (dim=-1 or dim=2 in 3D tensor)
// We assume the input is contiguous and we are reducing over the last axis.
// For a general dim, we can transpose/permute so that the reduction dimension is always the last one.

__global__ void argmin_last_dim_kernel(
    const float* __restrict__ input,
    int64_t* __restrict__ output,
    int batch_size,
    int outer_dims,
    int inner_dim
) {
    // Each thread block handles one element of the reduced dimension space (batch * outer_dims)
    // Total number of elements to process is batch_size * outer_dims
    
    int total_elements = batch_size * outer_dims;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        const float* row_ptr = input + idx * inner_dim;
        
        // Find min value and its index
        float min_val = row_ptr[0];
        int64_t min_idx = 0;

        for (int i = 1; i < inner_dim; ++i) {
            float val = row_ptr[i];
            if (val < min_val) {
                min_val = val;
                min_idx = i;
            }
        }
        
        output[idx] = min_idx;
    }
}

torch::Tensor argmin_cuda(torch::Tensor x, int dim) {
    // Ensure input is contiguous
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }

    auto sizes = x.sizes();
    int ndim = sizes.size();
    
    // Handle negative dimension
    if (dim < 0) {
        dim += ndim;
    }

    // Calculate the size of the reduction dimension and the rest
    int64_t reduce_size = sizes[dim];
    
    // Calculate number of elements before and after the reduction dimension
    int64_t outer_size = 1;
    for (int i = 0; i < dim; ++i) {
        outer_size *= sizes[i];
    }
    
    int64_t inner_size = 1;
    for (int i = dim + 1; i < ndim; ++i) {
        inner_size *= sizes[i];
    }

    // Output shape: all dimensions except the reduction dimension
    std::vector<int64_t> output_sizes;
    for (int i = 0; i < ndim; ++i) {
        if (i != dim) {
            output_sizes.push_back(sizes[i]);
        }
    }

    // Create output tensor with int64 type as argmin returns indices
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(x.device());
    torch::Tensor output = torch::empty(output_sizes, options);

    const int block_size = 256;
    int total_threads = outer_size * inner_size;
    int num_blocks = (total_threads + block_size - 1) / block_size;

    // Launch kernel
    argmin_last_dim_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<int64_t>(),
        outer_size,
        inner_size, // This is actually the product of dims after dim, but we need to map correctly
        reduce_size
    );

    return output;
}
"""

argmin_cpp_source = (
    "torch::Tensor argmin_cuda(torch::Tensor x, int dim);"
)

# Compile the inline CUDA code for argmin
argmin_module = load_inline(
    name="argmin_custom",
    cpp_sources=argmin_cpp_source,
    cuda_sources=argmin_source,
    functions=["argmin_cuda"],
    verbose=False,
    extra_cflags=["-O2"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Optimized model that finds the index of the minimum value along a specified dimension using custom CUDA.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to perform argmin on.

        Args:
            dim (int): Dimension along which to find the minimum value.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.argmin_func = argmin_module.argmin_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Finds the index of the minimum value along the specified dimension using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor containing the indices of the minimum values along the specified dimension.
        """
        return self.argmin_func(x, self.dim)