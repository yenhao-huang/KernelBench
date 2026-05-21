import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matvec_cpp_source = """
torch::Tensor matvec_cuda(torch::Tensor A, torch::Tensor B);
"""

matvec_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__inline__ __device__ float warp_sum(float v) {
    v += __shfl_down_sync(0xffffffff, v, 16);
    v += __shfl_down_sync(0xffffffff, v, 8);
    v += __shfl_down_sync(0xffffffff, v, 4);
    v += __shfl_down_sync(0xffffffff, v, 2);
    v += __shfl_down_sync(0xffffffff, v, 1);
    return v;
}

__inline__ __device__ float block_sum(float v) {
    __shared__ float smem[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    v = warp_sum(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();

    v = (threadIdx.x < (blockDim.x >> 5)) ? smem[lane] : 0.0f;
    if (wid == 0) v = warp_sum(v);
    return v;
}

__global__ void matvec_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M,
                              int K) {
    int row = blockIdx.x;
    if (row >= M) return;

    const float* a = A + ((long long)row * K);
    float sum = 0.0f;

    int K4 = K >> 2;
    const float4* __restrict__ a4 = reinterpret_cast<const float4*>(a);
    const float4* __restrict__ b4 = reinterpret_cast<const float4*>(B);

    for (int i = threadIdx.x; i < K4; i += blockDim.x) {
        float4 av = a4[i];
        float4 bv = b4[i];
        sum = fmaf(av.x, bv.x, sum);
        sum = fmaf(av.y, bv.y, sum);
        sum = fmaf(av.z, bv.z, sum);
        sum = fmaf(av.w, bv.w, sum);
    }

    for (int i = (K4 << 2) + threadIdx.x; i < K; i += blockDim.x) {
        sum = fmaf(a[i], B[i], sum);
    }

    sum = block_sum(sum);
    if (threadIdx.x == 0) {
        C[row] = sum;
    }
}

torch::Tensor matvec_cuda(torch::Tensor A, torch::Tensor B) {
    int M = (int)A.size(0);
    int K = (int)A.size(1);
    auto C = torch::empty({M, 1}, A.options());

    constexpr int threads = 256;
    matvec_kernel<<<M, threads>>>(A.data_ptr<float>(),
                                  B.data_ptr<float>(),
                                  C.data_ptr<float>(),
                                  M,
                                  K);
    return C;
}
"""

matvec_ext = load_inline(
    name="custom_matvec_fp32_ext",
    cpp_sources=matvec_cpp_source,
    cuda_sources=matvec_cuda_source,
    functions=["matvec_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matvec_ext = matvec_ext

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matvec_ext.matvec_cuda(A, B)