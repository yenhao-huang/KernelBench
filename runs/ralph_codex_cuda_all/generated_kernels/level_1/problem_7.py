import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul64_cpp_source = """
torch::Tensor matmul64_cuda(torch::Tensor A, torch::Tensor B);
"""

matmul64_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<int K>
__global__ void matmul_k64_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M,
                                  int N) {
    int col = blockIdx.x * 32 + threadIdx.x;
    int row = blockIdx.y * 8 + threadIdx.y;

    if (row < M && col < N) {
        float acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            acc += __ldg(A + row * K + k) * __ldg(B + k * N + col);
        }
        C[row * N + col] = acc;
    }
}

__global__ void matmul_generic_kernel(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C,
                                      int M,
                                      int N,
                                      int K) {
    int col = blockIdx.x * 32 + threadIdx.x;
    int row = blockIdx.y * 8 + threadIdx.y;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += __ldg(A + row * K + k) * __ldg(B + k * N + col);
        }
        C[row * N + col] = acc;
    }
}

torch::Tensor matmul64_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = (int)A.size(0);
    const int K = (int)A.size(1);
    const int N = (int)B.size(1);

    auto C = torch::empty({M, N}, A.options());

    dim3 block(32, 8);
    dim3 grid((N + 31) / 32, (M + 7) / 8);

    if (K == 64) {
        matmul_k64_kernel<64><<<grid, block>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            M,
            N
        );
    } else {
        matmul_generic_kernel<<<grid, block>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            M,
            N,
            K
        );
    }

    return C;
}
"""

matmul64_ext = load_inline(
    name="matmul64_ext",
    cpp_sources=matmul64_cpp_source,
    cuda_sources=matmul64_cuda_source,
    functions=["matmul64_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul64_ext = matmul64_ext

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul64_ext.matmul64_cuda(A, B)