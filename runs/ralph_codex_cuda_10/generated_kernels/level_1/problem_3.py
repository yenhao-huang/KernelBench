import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

bmm_cublas_cpp = r"""
#include <torch/extension.h>

torch::Tensor bmm_cublas_cuda(torch::Tensor A, torch::Tensor B);

torch::Tensor bmm_cublas(torch::Tensor A, torch::Tensor B) {
    return bmm_cublas_cuda(A, B);
}
"""

bmm_cublas_cuda = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

torch::Tensor bmm_cublas_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be CUDA");
    TORCH_CHECK(B.is_cuda(), "B must be CUDA");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.scalar_type() == torch::kFloat32, "B must be float32");

    A = A.contiguous();
    B = B.contiguous();

    const int64_t batch = A.size(0);
    const int64_t m = A.size(1);
    const int64_t k = A.size(2);
    const int64_t n = B.size(2);

    auto C = torch::empty({batch, m, n}, A.options());

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(handle, at::cuda::getCurrentCUDAStream());

    cublasStatus_t status = cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        (int)n,
        (int)m,
        (int)k,
        &alpha,
        B.data_ptr<float>(),
        (int)n,
        k * n,
        A.data_ptr<float>(),
        (int)k,
        m * k,
        &beta,
        C.data_ptr<float>(),
        (int)n,
        m * n,
        (int)batch
    );

    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cublasSgemmStridedBatched failed");
    return C;
}
"""

_bmm_cublas = load_inline(
    name="kb_bmm_cublas_fp32",
    cpp_sources=bmm_cublas_cpp,
    cuda_sources=bmm_cublas_cuda,
    functions=["bmm_cublas"],
    extra_ldflags=["-lcublas"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return _bmm_cublas.bmm_cublas(A, B)