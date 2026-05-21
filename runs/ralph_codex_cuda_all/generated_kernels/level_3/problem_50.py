import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void causal_relu_attention_kernel(
    const float* __restrict__ qkv,
    float* __restrict__ out,
    int B,
    int T,
    int C,
    int H
) {
    int hs = C / H;
    int tid = threadIdx.x;
    int block = blockIdx.x;

    int t = block % T;
    int h = (block / T) % H;
    int b = block / (T * H);

    extern __shared__ float sh[];

    float acc = 0.0f;
    int d = tid;
    float scale = rsqrtf((float)hs);

    if (d < hs) {
        int q_base = ((b * T + t) * (3 * C)) + h * hs;
        float qd = qkv[q_base + d];

        for (int j = 0; j <= t; ++j) {
            int k_base = ((b * T + j) * (3 * C)) + C + h * hs;
            sh[tid] = qd * qkv[k_base + d];
            __syncthreads();

            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    sh[tid] += sh[tid + stride];
                }
                __syncthreads();
            }

            float a = sh[0] * scale;
            a = a > 0.0f ? a : 0.0f;

            int v_base = ((b * T + j) * (3 * C)) + 2 * C + h * hs;
            acc += a * qkv[v_base + d];
            __syncthreads();
        }

        out[(b * T + t) * C + h * hs + d] = acc;
    }
}

torch::Tensor causal_relu_attention_cuda(torch::Tensor qkv, int64_t n_head) {
    int B = (int)qkv.size(0);
    int T = (int)qkv.size(1);
    int C = (int)qkv.size(2) / 3;
    int H = (int)n_head;
    int hs = C / H;

    auto out = torch::empty({B, T, C}, qkv.options());

    int threads = 1;
    while (threads < hs) threads <<= 1;
    if (threads < 32) threads = 32;

    dim3 grid(B * H * T);
    size_t smem = threads * sizeof(float);

    causal_relu_attention_kernel<<<grid, threads, smem>>>(
        qkv.data_ptr<float>(),
        out.data_ptr<float>(),
        B, T, C, H
    );

    return out;
}
"""

cpp_sources = r"""
#include <torch/extension.h>
torch::Tensor causal_relu_attention_cuda(torch::Tensor qkv, int64_t n_head);
"""

causal_relu_attention_ext = load_inline(
    name="causal_relu_attention_ext",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["causal_relu_attention_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self, n_embd, n_head, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_seqlen, max_seqlen)).view(1, 1, max_seqlen, max_seqlen),
        )
        self.n_head = n_head
        self.n_embd = n_embd
        self.attn_ext = causal_relu_attention_ext

    def forward(self, x):
        qkv = self.c_attn(x)
        return self.attn_ext.causal_relu_attention_cuda(qkv, self.n_head)