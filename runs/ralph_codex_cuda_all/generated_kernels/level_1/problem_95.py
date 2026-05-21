import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

ce_cpp_source = """
torch::Tensor cross_entropy_forward(torch::Tensor predictions, torch::Tensor targets);
"""

ce_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void cross_entropy_mean_kernel(
    const float* __restrict__ x,
    const long long* __restrict__ t,
    float* __restrict__ out,
    int n,
    int c
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float smax[256];
    __shared__ float ssum[256];

    const float* row_ptr = x + ((long long)row * c);

    float local_max = -FLT_MAX;
    for (int j = tid; j < c; j += blockDim.x) {
        float v = row_ptr[j];
        local_max = fmaxf(local_max, v);
    }

    smax[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smax[tid] = fmaxf(smax[tid], smax[tid + stride]);
        }
        __syncthreads();
    }

    float m = smax[0];
    float local_sum = 0.0f;
    for (int j = tid; j < c; j += blockDim.x) {
        local_sum += __expf(row_ptr[j] - m);
    }

    ssum[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            ssum[tid] += ssum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        int target = (int)t[row];
        float loss = (logf(ssum[0]) + m - row_ptr[target]) / (float)n;
        atomicAdd(out, loss);
    }
}

torch::Tensor cross_entropy_forward(torch::Tensor predictions, torch::Tensor targets) {
    int n = (int)predictions.size(0);
    int c = (int)predictions.size(1);

    auto out = torch::empty({}, predictions.options());
    cudaMemset(out.data_ptr<float>(), 0, sizeof(float));

    cross_entropy_mean_kernel<<<n, 256>>>(
        predictions.data_ptr<float>(),
        reinterpret_cast<const long long*>(targets.data_ptr<int64_t>()),
        out.data_ptr<float>(),
        n,
        c
    );

    return out;
}
"""

ce_ext = load_inline(
    name="cross_entropy_mean_inline_ext",
    cpp_sources=ce_cpp_source,
    cuda_sources=ce_cuda_source,
    functions=["cross_entropy_forward"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.ce_ext = ce_ext

    def forward(self, predictions, targets):
        return self.ce_ext.cross_entropy_forward(predictions, targets)