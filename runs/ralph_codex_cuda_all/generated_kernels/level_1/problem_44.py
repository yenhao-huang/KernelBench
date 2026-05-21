import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

avgpool1d_cpp_source = """
torch::Tensor avgpool1d_cuda(torch::Tensor x, int64_t kernel_size, int64_t stride, int64_t padding);
"""

avgpool1d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avgpool1d_k8_s1_kernel(const float* __restrict__ x,
                                       float* __restrict__ y,
                                       long long rows,
                                       int L,
                                       int O) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = rows * (long long)O;
    if (idx >= total) return;

    int o = (int)(idx % O);
    long long row = idx / O;
    const float* base = x + row * (long long)L;

    float sum;
    int s = o - 4;

    if (o >= 4 && o <= L - 4) {
        sum = base[s] + base[s + 1] + base[s + 2] + base[s + 3]
            + base[s + 4] + base[s + 5] + base[s + 6] + base[s + 7];
    } else {
        sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            int xi = s + k;
            if ((unsigned)xi < (unsigned)L) sum += base[xi];
        }
    }

    y[idx] = sum * 0.125f;
}

__global__ void avgpool1d_generic_kernel(const float* __restrict__ x,
                                         float* __restrict__ y,
                                         long long rows,
                                         int L,
                                         int O,
                                         int K,
                                         int S,
                                         int P) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = rows * (long long)O;
    if (idx >= total) return;

    int o = (int)(idx % O);
    long long row = idx / O;
    const float* base = x + row * (long long)L;
    int start = o * S - P;
    float sum = 0.0f;

    for (int k = 0; k < K; ++k) {
        int xi = start + k;
        if ((unsigned)xi < (unsigned)L) sum += base[xi];
    }

    y[idx] = sum / (float)K;
}

torch::Tensor avgpool1d_cuda(torch::Tensor x, int64_t kernel_size, int64_t stride, int64_t padding) {
    const int64_t N = x.size(0);
    const int64_t C = x.size(1);
    const int64_t L64 = x.size(2);
    const int64_t O64 = (L64 + 2 * padding - kernel_size) / stride + 1;

    auto y = torch::empty({N, C, O64}, x.options());

    const long long rows = (long long)N * (long long)C;
    const long long total = rows * (long long)O64;
    const int threads = 256;
    const int blocks = (int)((total + threads - 1) / threads);

    if (kernel_size == 8 && stride == 1 && padding == 4) {
        avgpool1d_k8_s1_kernel<<<blocks, threads>>>(
            x.data_ptr<float>(), y.data_ptr<float>(), rows, (int)L64, (int)O64
        );
    } else {
        avgpool1d_generic_kernel<<<blocks, threads>>>(
            x.data_ptr<float>(), y.data_ptr<float>(), rows, (int)L64, (int)O64,
            (int)kernel_size, (int)stride, (int)padding
        );
    }

    return y;
}
"""

avgpool1d_ext = load_inline(
    name="avgpool1d_inline_ext",
    cpp_sources=avgpool1d_cpp_source,
    cuda_sources=avgpool1d_cuda_source,
    functions=["avgpool1d_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.avgpool1d_ext = avgpool1d_ext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.avgpool1d_ext.avgpool1d_cuda(
            x.contiguous(), self.kernel_size, self.stride, self.padding
        )