import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

softmax_cpp_source = """
torch::Tensor softmax_dim1_cuda(torch::Tensor x);
"""

softmax_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

template<int BLOCK>
__global__ void softmax_dim1_kernel(const float* __restrict__ x,
                                    float* __restrict__ out,
                                    int rows,
                                    int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_x = x + (long long)row * cols;
    float* row_out = out + (long long)row * cols;

    float local_max = -FLT_MAX;
    for (int c = threadIdx.x; c < cols; c += BLOCK) {
        float v = row_x[c];
        local_max = fmaxf(local_max, v);
    }

    __shared__ float smem[BLOCK];
    smem[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = BLOCK >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    float max_val = smem[0];
    float local_sum = 0.0f;

    for (int c = threadIdx.x; c < cols; c += BLOCK) {
        local_sum += expf(row_x[c] - max_val);
    }

    smem[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = BLOCK >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float inv_sum = 1.0f / smem[0];

    for (int c = threadIdx.x; c < cols; c += BLOCK) {
        row_out[c] = expf(row_x[c] - max_val) * inv_sum;
    }
}

torch::Tensor softmax_dim1_cuda(torch::Tensor x) {
    auto out = torch::empty_like(x);
    int rows = (int)x.size(0);
    int cols = (int)x.size(1);

    constexpr int BLOCK = 256;
    softmax_dim1_kernel<BLOCK><<<rows, BLOCK>>>(x.data_ptr<float>(), out.data_ptr<float>(), rows, cols);

    return out;
}
"""

softmax_ext = load_inline(
    name="softmax_dim1_kernelbench_ext",
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_cuda_source,
    functions=["softmax_dim1_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softmax_ext = softmax_ext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax_ext.softmax_dim1_cuda(x.contiguous())