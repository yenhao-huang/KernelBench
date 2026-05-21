import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused scale, mask, and ReLU
scaled_masked_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void scaled_masked_relu_kernel(
    const float* __restrict__ att,
    const float* __restrict__ mask,
    float* __restrict__ out,
    float scale,
    int B,
    int nh,
    int T,
    int max_seqlen)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * nh * T * T;
    if (idx < total) {
        int tmp = idx;
        int j = tmp % T; tmp /= T;
        int i = tmp % T; tmp /= T;
        int h = tmp % nh; tmp /= nh;
        int b = tmp;

        // mask is (1, 1, max_seqlen, max_seqlen), contiguous
        float m = mask[i * max_seqlen + j];
        if (m == 0.0f) {
            out[idx] = 0.0f;
        } else {
            float val = att[idx] * scale;
            out[idx] = val > 0.0f ? val : 0.0f;
        }
    }
}

torch::Tensor scaled_masked_relu_cuda(
    torch::Tensor att,
    torch::Tensor mask,
    float scale)
{
    int B = att.size(0);
    int nh = att.size(1);
    int T = att.size(2);
    int max_seqlen = mask.size(2); // mask shape (1,1,max_seqlen,max_seqlen)

    auto out = torch::empty_like(att);

    int total = B * nh * T * T;
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;

    scaled_masked_relu_kernel<<<num_blocks, block_size>>>(
        att.data_ptr<float>(),
        mask.data_ptr<float>(),
        out.data_ptr<float>(),
        scale,
        B,
        nh,
        T,
        max_seqlen
    );

    return out;
}
"""

scaled_masked_relu_cpp_source = (
    "torch::Tensor scaled_masked_relu_cuda(torch::Tensor att, torch::Tensor mask, float scale);"
)

# Compile the inline CUDA code
scaled_masked_relu = load_inline(
    name="scaled_masked_relu",
    cpp_sources=scaled_masked_relu_cpp_source,
    cuda_sources=scaled_masked_relu_source,
    functions=["scaled_masked_relu_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super(NewGELU, self).__init__()
    
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class ModelNew(nn.Module):
    """
    A multi-head masked self-attention layer with a projection at the end that uses ReLU instead of Softmax.
    Optimized with a custom CUDA kernel that fuses scaling, causal masking, and ReLU.
    """

    def __init__(self, n_embd, n_head, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                                     .view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd
        self.scaled_masked_relu = scaled_masked_relu

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # raw attention scores: (B, nh, T, T)
        raw_att = q @ k.transpose(-2, -1)
        scale = 1.0 / math.sqrt(k.size(-1))

        # Fused scale, mask, and ReLU
        att = self.scaled_masked_relu.scaled_masked_relu_cuda(raw_att, self.bias, scale)

        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        return y


batch_size = 16
max_seqlen = 1024
n_embd = 768  # Hidden dimension, typical for BERT-base size
n_head = 12   # Number of attention heads, typical for BERT-base size


def get_inputs():
    return [torch.rand(batch_size, max_seqlen, n_embd)]


def get_init_inputs():
    return [n_embd, n_head, max_seqlen]