import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_cublas_cpp = r"""
#include <torch/extension.h>

torch::Tensor matmul_cublas_cuda(torch::Tensor A, torch::Tensor B);

torch::Tensor matmul_cublas(torch::Tensor A, torch::Tensor B) {
    return matmul_cublas_cuda(A, B);
}
"""

matmul_cublas_cuda = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>

torch::Tensor matmul_cublas_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 && B.scalar_type() == torch::kFloat32, "A and B must be float32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(1) == B.size(0), "incompatible matrix sizes");

    c10::cuda::CUDAGuard device_guard(A.device());

    auto Ac = A.contiguous();
    auto Bc = B.contiguous();
    auto C = torch::empty({A.size(0), B.size(1)}, A.options());

    const int n = (int)A.size(0);
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cublasSetStream(handle, stream);

    cublasStatus_t status = cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n,
        n,
        n,
        &alpha,
        Bc.data_ptr<float>(),
        n,
        Ac.data_ptr<float>(),
        n,
        &beta,
        C.data_ptr<float>(),
        n
    );

    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cublasSgemm failed");
    return C;
}
"""

_matmul_cublas = load_inline(
    name="kernelbench_matmul_cublas_fp32",
    cpp_sources=matmul_cublas_cpp,
    cuda_sources=matmul_cublas_cuda,
    functions=["matmul_cublas"],
    extra_cuda_cflags=["-O3"],
    extra_cflags=["-O3"],
    extra_ldflags=["-lcublas"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._op = _matmul_cublas

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if not A.is_cuda or not B.is_cuda:
            return torch.matmul(A, B)
        return self._op.matmul_cublas(A, B)