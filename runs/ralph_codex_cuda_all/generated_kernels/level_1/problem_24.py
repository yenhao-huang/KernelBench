import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

logsoftmax_cpp_source = """
torch::Tensor logsoftmax_cuda(torch::Tensor x);
"""

logsoftmax_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void logsoftmax_rows_kernel(const float* __restrict__ x,
                                       float* __restrict__ y,
                                       int rows,
                                       int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_x = x + (long long)row * cols;
    float* row_y = y + (long long)row * cols;

    float local_max = -FLT_MAX;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = row_x[c];
        local_max = fmaxf(local_max, v);
    }

    __shared__ float smem[256];
    smem[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    float max_val = smem[0];

    float local_sum = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        local_sum += expf(row_x[c] - max_val);
    }

    smem[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float log_sum = logf(smem[0]);

    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        row_y[c] = row_x[c] - max_val - log_sum;
    }
}

torch::Tensor logsoftmax_cuda(torch::Tensor x) {
    auto y = torch::empty_like(x);
    int rows = x.size(0);
    int cols = x.size(1);
    dim3 block(256);
    dim3 grid(rows);
    logsoftmax_rows_kernel<<<grid, block>>>(x.data_ptr<float>(), y.data_ptr<float>(), rows, cols);
    return y;
}
"""

logsoftmax_ext = load_inline(
    name="kb_logsoftmax_ext",
    cpp_sources=logsoftmax_cpp_source,
    cuda_sources=logsoftmax_cuda_source,
    functions=["logsoftmax_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim
        self.logsoftmax_ext = logsoftmax_ext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logsoftmax_ext.logsoftmax_cuda(x)