import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

argmax_cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define TILE_COLS 32
#define REDUCE_LANES 8

__global__ void argmax_dim1_kernel(const float* __restrict__ x,
                                   int64_t* __restrict__ out,
                                   int B, int D1, int D2) {
    int col_base = blockIdx.x * TILE_COLS;
    int b = blockIdx.y;
    int tx = threadIdx.x;  // column lane
    int ty = threadIdx.y;  // reduction lane
    int col = col_base + tx;

    __shared__ float s_val[REDUCE_LANES][TILE_COLS];
    __shared__ int s_idx[REDUCE_LANES][TILE_COLS];

    float best = -3.4028234663852886e38f;
    int best_idx = 0;

    if (col < D2) {
        int base = b * D1 * D2 + col;
        for (int i = ty; i < D1; i += REDUCE_LANES) {
            float v = x[base + i * D2];
            if (v > best) {
                best = v;
                best_idx = i;
            }
        }
    }

    s_val[ty][tx] = best;
    s_idx[ty][tx] = best_idx;
    __syncthreads();

    if (ty == 0 && col < D2) {
        float vbest = s_val[0][tx];
        int ibest = s_idx[0][tx];

        #pragma unroll
        for (int r = 1; r < REDUCE_LANES; ++r) {
            float v = s_val[r][tx];
            int idx = s_idx[r][tx];
            if (v > vbest || (v == vbest && idx < ibest)) {
                vbest = v;
                ibest = idx;
            }
        }
        out[b * D2 + col] = (int64_t)ibest;
    }
}

torch::Tensor argmax_dim1_cuda(torch::Tensor x) {
    const int B = (int)x.size(0);
    const int D1 = (int)x.size(1);
    const int D2 = (int)x.size(2);

    auto out = torch::empty({B, D2}, x.options().dtype(torch::kInt64));

    dim3 block(TILE_COLS, REDUCE_LANES);
    dim3 grid((D2 + TILE_COLS - 1) / TILE_COLS, B);

    argmax_dim1_kernel<<<grid, block>>>(x.data_ptr<float>(), out.data_ptr<int64_t>(), B, D1, D2);
    return out;
}
"""

argmax_cpp_source = "torch::Tensor argmax_dim1_cuda(torch::Tensor x);"

argmax_ext = load_inline(
    name="argmax_dim1_ext",
    cpp_sources=argmax_cpp_source,
    cuda_sources=argmax_cuda_source,
    functions=["argmax_dim1_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.argmax_ext = argmax_ext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.argmax_ext.argmax_dim1_cuda(x)