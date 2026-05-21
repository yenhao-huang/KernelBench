import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA source code for min reduction along any dimension for 3D tensor
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define FLOAT_INF __int_as_float(0x7f800000)

template<int DIM>
__global__ void min_reduce_kernel(const float* __restrict__ input, float* __restrict__ output,
                                  int dim0, int dim1, int dim2,
                                  int stride0, int stride1, int stride2,
                                  int red_size, int out_size) {
    // Compute output linear index from grid
    int out_idx = blockIdx.x + blockIdx.y * gridDim.x;
    if (out_idx >= out_size) return;

    // Determine non-reduction indices
    int idx0, idx1, idx2;
    if (DIM == 0) {
        idx1 = out_idx / dim2;
        idx2 = out_idx % dim2;
    } else if (DIM == 1) {
        idx0 = out_idx / dim2;
        idx2 = out_idx % dim2;
    } else { // DIM == 2
        idx0 = out_idx / dim1;
        idx1 = out_idx % dim1;
    }

    float min_val = FLOAT_INF;

    // Thread-local reduction loop
    for (int i = threadIdx.x; i < red_size; i += blockDim.x) {
        int input_idx;
        if (DIM == 0) {
            input_idx = i * stride0 + idx1 * stride1 + idx2 * stride2;
        } else if (DIM == 1) {
            input_idx = idx0 * stride0 + i * stride1 + idx2 * stride2;
        } else {
            input_idx = idx0 * stride0 + idx1 * stride1 + i * stride2;
        }
        float val = input[input_idx];
        if (val < min_val) min_val = val;
    }

    // Shared memory reduction
    extern __shared__ float sdata[];
    sdata[threadIdx.x] = min_val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float other = sdata[threadIdx.x + stride];
            if (other < sdata[threadIdx.x]) sdata[threadIdx.x] = other;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[out_idx] = sdata[0];
    }
}

torch::Tensor min_reduce_dim0_cuda(torch::Tensor x) {
    auto sizes = x.sizes();
    int B = sizes[0], M = sizes[1], N = sizes[2];
    auto strides = x.strides();
    int out_size = M * N;
    auto out = torch::empty({M, N}, x.options());

    int block_size = 256;
    // Use 2D grid to handle large out_size if needed
    dim3 grid(std::min(out_size, 65535), (out_size + 65534) / 65535, 1);

    min_reduce_kernel<0><<<grid, block_size, block_size * sizeof(float)>>>(
        x.data_ptr<float>(), out.data_ptr<float>(),
        B, M, N,
        strides[0], strides[1], strides[2],
        B, out_size);
    return out;
}

torch::Tensor min_reduce_dim1_cuda(torch::Tensor x) {
    auto sizes = x.sizes();
    int B = sizes[0], M = sizes[1], N = sizes[2];
    auto strides = x.strides();
    int out_size = B * N;
    auto out = torch::empty({B, N}, x.options());

    int block_size = 256;
    dim3 grid(std::min(out_size, 65535), (out_size + 65534) / 65535, 1);

    min_reduce_kernel<1><<<grid, block_size, block_size * sizeof(float)>>>(
        x.data_ptr<float>(), out.data_ptr<float>(),
        B, M, N,
        strides[0], strides[1], strides[2],
        M, out_size);
    return out;
}

torch::Tensor min_reduce_dim2_cuda(torch::Tensor x) {
    auto sizes = x.sizes();
    int B = sizes[0], M = sizes[1], N = sizes[2];
    auto strides = x.strides();
    int out_size = B * M;
    auto out = torch::empty({B, M}, x.options());

    int block_size = 256;
    dim3 grid(std::min(out_size, 65535), (out_size + 65534) / 65535, 1);

    min_reduce_kernel<2><<<grid, block_size, block_size * sizeof(float)>>>(
        x.data_ptr<float>(), out.data_ptr<float>(),
        B, M, N,
        strides[0], strides[1], strides[2],
        N, out_size);
    return out;
}
"""

cpp_source = """
torch::Tensor min_reduce_dim0_cuda(torch::Tensor x);
torch::Tensor min_reduce_dim1_cuda(torch::Tensor x);
torch::Tensor min_reduce_dim2_cuda(torch::Tensor x);
"""

# Compile the inline CUDA code
min_reduce_ops = load_inline(
    name="min_reduce_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["min_reduce_dim0_cuda", "min_reduce_dim1_cuda", "min_reduce_dim2_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.min_reduce_ops = min_reduce_ops

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous for memory coalescing
        x = x.contiguous()
        if self.dim == 0:
            return self.min_reduce_ops.min_reduce_dim0_cuda(x)
        elif self.dim == 1:
            return self.min_reduce_ops.min_reduce_dim1_cuda(x)
        elif self.dim == 2:
            return self.min_reduce_ops.min_reduce_dim2_cuda(x)
        else:
            raise ValueError("Dimension out of range for 3D tensor")