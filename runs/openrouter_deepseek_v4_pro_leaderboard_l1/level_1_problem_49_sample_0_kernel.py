import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for max reduction along a dimension
max_reduce_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void max_reduce_kernel(const float* input, float* output,
                                  int dim,
                                  int size0, int size1, int size2,
                                  int stride0, int stride1, int stride2,
                                  int out_size0, int out_size1,
                                  int out_stride0, int out_stride1,
                                  int reduce_size, int reduce_stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = out_size0 * out_size1;
    if (idx < total_out) {
        int out_idx0 = idx / out_size1;
        int out_idx1 = idx % out_size1;
        int base_offset;
        if (dim == 0) {
            base_offset = out_idx0 * stride1 + out_idx1 * stride2;
        } else if (dim == 1) {
            base_offset = out_idx0 * stride0 + out_idx1 * stride2;
        } else { // dim == 2
            base_offset = out_idx0 * stride0 + out_idx1 * stride1;
        }
        float max_val = -FLT_MAX;
        for (int i = 0; i < reduce_size; ++i) {
            float val = input[base_offset + i * reduce_stride];
            if (val > max_val) max_val = val;
        }
        output[idx] = max_val;
    }
}

torch::Tensor max_reduce_cuda(torch::Tensor x, int dim) {
    // Ensure input is contiguous
    x = x.contiguous();
    auto sizes = x.sizes();
    auto strides = x.strides();
    int size0 = sizes[0], size1 = sizes[1], size2 = sizes[2];
    int stride0 = strides[0], stride1 = strides[1], stride2 = strides[2];
    int reduce_size = sizes[dim];
    int reduce_stride = strides[dim];

    // Compute output shape (remove the reduction dimension)
    std::vector<int64_t> out_shape;
    for (int i = 0; i < 3; ++i) {
        if (i != dim) out_shape.push_back(sizes[i]);
    }
    auto out = torch::empty(out_shape, x.options());

    int out_size0 = out_shape[0];
    int out_size1 = out_shape[1];
    int out_stride0 = out.strides()[0];
    int out_stride1 = out.strides()[1];
    int total_out = out_size0 * out_size1;

    const int block_size = 256;
    const int num_blocks = (total_out + block_size - 1) / block_size;

    max_reduce_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), out.data_ptr<float>(),
        dim,
        size0, size1, size2,
        stride0, stride1, stride2,
        out_size0, out_size1,
        out_stride0, out_stride1,
        reduce_size, reduce_stride
    );

    return out;
}
"""

max_reduce_cpp_source = "torch::Tensor max_reduce_cuda(torch::Tensor x, int dim);"

# Compile the inline CUDA code
max_reduce = load_inline(
    name="max_reduce",
    cpp_sources=max_reduce_cpp_source,
    cuda_sources=max_reduce_source,
    functions=["max_reduce_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs Max reduction over a specific dimension using a custom CUDA kernel.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.max_reduce = max_reduce

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max reduction over the specified dimension to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after Max reduction over the specified dimension.
        """
        return self.max_reduce.max_reduce_cuda(x, self.dim)