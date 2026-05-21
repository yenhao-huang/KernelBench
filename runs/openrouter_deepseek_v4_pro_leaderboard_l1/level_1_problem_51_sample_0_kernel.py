import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA source for argmax
argmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define MAX_NDIM 16

__global__ void argmax_kernel(
    const float* __restrict__ input,
    int64_t* __restrict__ output,
    int ndim,
    const int64_t* __restrict__ shape,
    const int64_t* __restrict__ strides,
    const int64_t* __restrict__ out_shape,
    int64_t out_numel,
    int64_t reduce_dim,
    int64_t reduce_dim_size,
    int64_t reduce_dim_stride
) {
    int out_idx = blockIdx.x;
    if (out_idx >= out_numel) return;

    // compute base offset for this output element
    int64_t out_indices[MAX_NDIM];
    int64_t rem = out_idx;
    for (int d = ndim - 2; d >= 0; --d) {
        out_indices[d] = rem % out_shape[d];
        rem /= out_shape[d];
    }

    // form input indices by inserting 0 at reduce_dim
    int64_t in_indices[MAX_NDIM];
    int out_d = 0;
    for (int d = 0; d < ndim; ++d) {
        if (d == reduce_dim) {
            in_indices[d] = 0;
        } else {
            in_indices[d] = out_indices[out_d++];
        }
    }

    // compute base offset in input
    int64_t base_offset = 0;
    for (int d = 0; d < ndim; ++d) {
        base_offset += in_indices[d] * strides[d];
    }

    // perform block reduction to find argmax along reduction dimension
    __shared__ float s_max[256];
    __shared__ int64_t s_idx[256];

    int tid = threadIdx.x;
    float max_val = -1e30f;
    int64_t max_idx = -1;

    for (int64_t k = tid; k < reduce_dim_size; k += blockDim.x) {
        float val = input[base_offset + k * reduce_dim_stride];
        if (val > max_val) {
            max_val = val;
            max_idx = k;
        }
    }

    s_max[tid] = max_val;
    s_idx[tid] = max_idx;
    __syncthreads();

    // tree reduction in shared memory
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            float other_val = s_max[tid + offset];
            if (other_val > s_max[tid]) {
                s_max[tid] = other_val;
                s_idx[tid] = s_idx[tid + offset];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[out_idx] = s_idx[0];
    }
}

torch::Tensor argmax_cuda(torch::Tensor input, int64_t dim) {
    // normalize dim
    if (dim < 0) dim += input.ndimension();
    int ndim = input.ndimension();

    TORCH_CHECK(ndim <= MAX_NDIM, "Number of dimensions exceeds maximum supported (", MAX_NDIM, ")");

    auto shape = input.sizes().vec();
    auto strides = input.strides().vec();

    // build output shape (without reduction dimension)
    std::vector<int64_t> out_shape_vec;
    for (int i = 0; i < ndim; ++i) {
        if (i != dim) {
            out_shape_vec.push_back(shape[i]);
        }
    }

    auto output = torch::empty(out_shape_vec, torch::dtype(torch::kInt64).device(input.device()));

    int64_t out_numel = output.numel();
    if (out_numel == 0) {
        return output;
    }

    int64_t reduce_dim_size = shape[dim];
    int64_t reduce_dim_stride = strides[dim];

    // copy shape/strides arrays to device as tensors
    auto shape_tensor = torch::tensor(shape, torch::TensorOptions().dtype(torch::kInt64).device(input.device()));
    auto strides_tensor = torch::tensor(strides, torch::TensorOptions().dtype(torch::kInt64).device(input.device()));
    auto out_shape_tensor = torch::tensor(out_shape_vec, torch::TensorOptions().dtype(torch::kInt64).device(input.device()));

    const int block_size = 256;
    int grid_size = out_numel;

    argmax_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<int64_t>(),
        ndim,
        shape_tensor.data_ptr<int64_t>(),
        strides_tensor.data_ptr<int64_t>(),
        out_shape_tensor.data_ptr<int64_t>(),
        out_numel,
        dim,
        reduce_dim_size,
        reduce_dim_stride
    );

    return output;
}
"""

argmax_cpp_source = "torch::Tensor argmax_cuda(torch::Tensor input, int64_t dim);"

# Compile the inline CUDA code
argmax_op = load_inline(
    name="argmax_cuda",
    cpp_sources=argmax_cpp_source,
    cuda_sources=argmax_source,
    functions=["argmax_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA argmax operator.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to perform argmax.

        Args:
            dim (int): The dimension to perform argmax over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.argmax = argmax_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies argmax over the specified dimension using a custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with argmax applied, with the specified dimension removed.
        """
        return self.argmax.argmax_cuda(x, self.dim)