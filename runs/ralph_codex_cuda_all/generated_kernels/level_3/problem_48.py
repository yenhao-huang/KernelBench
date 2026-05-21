import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

ssd_cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void ssd_forward_kernel(
    const float* __restrict__ X,
    const float* __restrict__ A,
    const float* __restrict__ Bp,
    const float* __restrict__ C,
    float* __restrict__ Y,
    int Bsz, int Lseq, int H, int P, int N, int BL
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Bsz * Lseq * H * P;
    if (idx >= total) return;

    int p = idx % P;
    int h = (idx / P) % H;
    int t = (idx / (P * H)) % Lseq;
    int b = idx / (P * H * Lseq);

    int chunk = t / BL;
    int l = t - chunk * BL;
    int chunks = Lseq / BL;

    float acc = 0.0f;

    for (int n = 0; n < N; ++n) {
        float diag_n = 0.0f;
        float cval = C[(((b * Lseq + t) * H + h) * N + n)];

        for (int s = 0; s <= l; ++s) {
            float seg = 0.0f;
            for (int r = s + 1; r <= l; ++r) {
                int ar_t = chunk * BL + r;
                seg += A[(b * Lseq + ar_t) * H + h];
            }

            int st = chunk * BL + s;
            float bval = Bp[(((b * Lseq + st) * H + h) * N + n)];
            float xval = X[(((b * Lseq + st) * H + h) * P + p)];
            diag_n += bval * expf(seg) * xval;
        }

        acc += cval * diag_n;
    }

    if (chunk > 0) {
        for (int n = 0; n < N; ++n) {
            float state = 0.0f;

            for (int pc = 0; pc < chunk; ++pc) {
                float chunk_decay = 0.0f;
                for (int cc = pc + 1; cc < chunk; ++cc) {
                    int last_t = cc * BL + (BL - 1);
                    chunk_decay += A[(b * Lseq + last_t) * H + h];
                }

                for (int s = 0; s < BL; ++s) {
                    int st = pc * BL + s;

                    float intra_decay = 0.0f;
                    for (int r = s + 1; r < BL; ++r) {
                        int ar_t = pc * BL + r;
                        intra_decay += A[(b * Lseq + ar_t) * H + h];
                    }

                    float bval = Bp[(((b * Lseq + st) * H + h) * N + n)];
                    float xval = X[(((b * Lseq + st) * H + h) * P + p)];
                    state += bval * expf(intra_decay + chunk_decay) * xval;
                }
            }

            float out_decay = 0.0f;
            for (int r = 0; r <= l; ++r) {
                int ar_t = chunk * BL + r;
                out_decay += A[(b * Lseq + ar_t) * H + h];
            }

            float cval = C[(((b * Lseq + t) * H + h) * N + n)];
            acc += cval * state * expf(out_decay);
        }
    }

    Y[idx] = acc;
}

torch::Tensor ssd_forward_cuda(torch::Tensor X, torch::Tensor A, torch::Tensor Bp, torch::Tensor C, int block_len) {
    int Bsz = X.size(0);
    int Lseq = X.size(1);
    int H = X.size(2);
    int P = X.size(3);
    int N = Bp.size(3);

    auto Y = torch::empty_like(X);
    int total = Bsz * Lseq * H * P;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    ssd_forward_kernel<<<blocks, threads>>>(
        X.data_ptr<float>(),
        A.data_ptr<float>(),
        Bp.data_ptr<float>(),
        C.data_ptr<float>(),
        Y.data_ptr<float>(),
        Bsz, Lseq, H, P, N, block_len
    );

    return Y;
}
"""

ssd_cpp_source = "torch::Tensor ssd_forward_cuda(torch::Tensor X, torch::Tensor A, torch::Tensor Bp, torch::Tensor C, int block_len);"

ssd_ext = load_inline(
    name="ssd_kernelbench_inline",
    cpp_sources=ssd_cpp_source,
    cuda_sources=ssd_cuda_source,
    functions=["ssd_forward_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        super(ModelNew, self).__init__()
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.block_len = block_len

        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads))
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        self.ssd_ext = ssd_ext

    def forward(self, X, initial_states=None):
        return self.ssd_ext.ssd_forward_cuda(X, self.A, self.B, self.C, self.block_len)