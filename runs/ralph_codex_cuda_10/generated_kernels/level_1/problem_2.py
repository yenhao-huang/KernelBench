import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_cpp = r"""
torch::Tensor matmul_cublas_cuda(torch::Tensor A, torch::Tensor B);
"""

matmul_cuda = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>

torch::Tensor matmul_cublas_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32, "A and B must be float32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "incompatible matmul dimensions");

    if (!A.is_contiguous()) A = A.contiguous();
    if (!B.is_contiguous()) B = B.contiguous();

    const int64_t M64 = A.size(0);
    const int64_t K64 = A.size(1);
    const int64_t N64 = B.size(1);

    TORCH_CHECK(M64 <= INT_MAX && K64 <= INT_MAX && N64 <= INT_MAX, "dimensions exceed cuBLAS int range");

    const int M = static_cast<int>(M64);
    const int K = static_cast<int>(K64);
    const int N = static_cast<int>(N64);

    auto C = torch::empty({M64, N64}, A.options());

    const c10::cuda::CUDAGuard device_guard(A.device());
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasStatus_t status = cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N,
        M,
        K,
        &alpha,
        B.data_ptr<float>(),
        N,
        A.data_ptr<float>(),
        K,
        &beta,
        C.data_ptr<float>(),
        N
    );

    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cublasSgemm failed");
    return C;
}
"""

_matmul_ext = load_inline(
    name="kb_cublas_matmul_fp32",
    cpp_sources=matmul_cpp,
    cuda_sources=matmul_cuda,
    functions=["matmul_cublas_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    extra_ldflags=["-lcublas"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if A.is_cuda and B.is_cuda and A.dtype == torch.float32 and B.dtype == torch.float32:
            return _matmul_ext.matmul_cublas_cuda(A, B)
        return torch.matmul(A, B)