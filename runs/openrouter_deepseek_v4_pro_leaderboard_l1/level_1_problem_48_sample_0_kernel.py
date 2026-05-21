import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for mean reduction along a specified dimension (3D tensor)
mean_reduce_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mean_reduce_kernel(const float* input, float* output,
                                   int dim0, int dim1, int dim2, int reduce_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_size;

    if (reduce_dim == 0) {
        out_size = dim1 * dim2;
        if (idx < out_size) {
            int i = idx / dim2;  // index in dim1
            int j = idx % dim2;  // index in dim2
            float sum = 0.0f;
            for (int k = 0; k < dim0; ++k) {
                sum += input[k * dim1 * dim2 + i * dim2 + j];
            }
            output[idx] = sum / dim0;
        }
    } else if (reduce_dim == 1) {
        out_size = dim0 * dim2;
        if (idx < out_size) {
            int i = idx / dim2;  // index in dim0
            int j = idx % dim2;  // index in dim2
            float sum = 0.0f;
            for (int k = 0; k < dim1; ++k) {
                sum += input[i * dim1 * dim2 + k * dim2 + j];
            }
            output[idx] = sum / dim1;
        }
    } else if (reduce_dim == 2) {
        out_size = dim0 * dim1;
        if (idx < out_size) {
            int i = idx / dim1;  // index in dim0
            int j = idx % dim1;  // index in dim1
            float sum = 0.0f;
            for (int k = 0; k < dim2; ++k) {
                sum += input[i * dim1 * dim2 + j * dim2 + k];
            }
            output[idx] = sum / dim2;
        }
    }
}

torch::Tensor mean_reduce_cuda(torch::Tensor input, int dim) {
    // Assume input is a 3D contiguous tensor
    auto sizes = input.sizes();
    int dim0 = sizes[0];
    int dim1 = sizes[1];
    int dim2 = sizes[2];

    // Compute output shape by removing the reduced dimension
    std::vector<int64_t> out_shape;
    for (int d = 0; d < 3; ++d) {
        if (d != dim) out_shape.push_back(sizes[d]);
    }
    auto output = torch::zeros(out_shape, input.options());

    int out_numel = output.numel();
    const int block_size = 256;
    const int num_blocks = (out_numel + block_size - 1) / block_size;

    mean_reduce_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        dim0, dim1, dim2, dim);

    return output;
}
"""

mean_reduce_cpp_source = "torch::Tensor mean_reduce_cuda(torch::Tensor input, int dim);"

# Compile the inline CUDA code
mean_reduce_op = load_inline(
    name="mean_reduce_op",
    cpp_sources=mean_reduce_cpp_source,
    cuda_sources=mean_reduce_cuda_source,
    functions=["mean_reduce_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.mean_reduce = mean_reduce_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mean_reduce.mean_reduce_cuda(x, self.dim)