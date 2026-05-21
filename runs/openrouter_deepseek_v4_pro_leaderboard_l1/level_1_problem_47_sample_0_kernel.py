import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for sum reduction along an arbitrary dimension (keepdim=True)
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define MAX_NDIM 8

__constant__ int64_t d_out_shape[MAX_NDIM];
__constant__ int64_t d_in_strides[MAX_NDIM];
__constant__ int64_t d_ndim;
__constant__ int64_t d_dim;

__global__ void sum_reduce_kernel(float* input, float* output, int64_t reduce_size, int64_t outer_size) {
    __shared__ float sdata[BLOCK_SIZE];

    int64_t linear_idx = blockIdx.x;
    if (linear_idx >= outer_size) return;

    // Unflatten the linear output index into multi-dimensional coordinates
    int64_t coord[MAX_NDIM];
    int64_t residual = linear_idx;
    for (int d = d_ndim - 1; d >= 0; --d) {
        coord[d] = residual % d_out_shape[d];
        residual /= d_out_shape[d];
    }

    int64_t base_offset = 0;
    for (int d = 0; d < d_ndim; ++d) {
        if (d != d_dim) {
            base_offset += coord[d] * d_in_strides[d];
        }
    }

    int64_t stride_dim = d_in_strides[d_dim];

    // Each thread accumulates its chunk of the reduction dimension
    float partial = 0.0f;
    int tid = threadIdx.x;
    for (int64_t i = tid; i < reduce_size; i += BLOCK_SIZE) {
        partial += input[base_offset + i * stride_dim];
    }
    sdata[tid] = partial;
    __syncthreads();

    // Tree reduction inside the block
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the result
    if (tid == 0) {
        output[linear_idx] = sdata[0];
    }
}

torch::Tensor custom_sum_cuda(torch::Tensor input, int64_t dim) {
    input = input.contiguous();
    AT_ASSERTM(input.is_cuda(), "Input must be a CUDA tensor");
    AT_ASSERTM(input.dtype() == torch::kFloat32, "Input must be float32");

    int64_t ndim = input.dim();
    AT_ASSERTM(dim >= 0 && dim < ndim, "Dimension out of range");
    AT_ASSERTM(ndim <= MAX_NDIM, "Too many dimensions");

    // Build output shape (keepdim=True)
    std::vector<int64_t> output_sizes(ndim);
    for (int64_t d = 0; d < ndim; ++d) {
        output_sizes[d] = (d == dim) ? 1 : input.size(d);
    }

    auto output = torch::empty(output_sizes, input.options());

    int64_t reduce_size = input.size(dim);
    int64_t outer_size = output.numel();

    // Extract strides and shape for constant memory
    std::vector<int64_t> in_strides(ndim);
    for (int64_t d = 0; d < ndim; ++d) {
        in_strides[d] = input.stride(d);
    }
    std::vector<int64_t> out_shape(ndim);
    for (int64_t d = 0; d < ndim; ++d) {
        out_shape[d] = output_sizes[d];
    }

    // Copy to device constant memory
    cudaMemcpyToSymbol(d_out_shape, out_shape.data(), ndim * sizeof(int64_t));
    cudaMemcpyToSymbol(d_in_strides, in_strides.data(), ndim * sizeof(int64_t));
    cudaMemcpyToSymbol(d_ndim, &ndim, sizeof(int64_t));
    cudaMemcpyToSymbol(d_dim, &dim, sizeof(int64_t));

    float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    dim3 block(BLOCK_SIZE);
    dim3 grid(outer_size);

    sum_reduce_kernel<<<grid, block>>>(input_ptr, output_ptr, reduce_size, outer_size);
    cudaDeviceSynchronize();
    AT_ASSERTM(cudaGetLastError() == cudaSuccess, "Kernel launch failed");

    return output;
}
"""

cpp_source = "torch::Tensor custom_sum_cuda(torch::Tensor input, int64_t dim);"

# Compile the inline CUDA code
custom_sum_module = load_inline(
    name="custom_sum_reduce",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["custom_sum_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.custom_sum = custom_sum_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Assumes input is already on CUDA and is float32.
        return self.custom_sum.custom_sum_cuda(x, self.dim)