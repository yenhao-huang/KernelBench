import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for efficient segment sum exponential
segsum_exp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void segsum_exp_kernel(const float* x, float* out, int T, int total_elements) {
    int elem_idx = blockIdx.x;
    if (elem_idx >= total_elements) return;

    const float* x_elem = x + elem_idx * T;
    float* out_elem = out + elem_idx * T * T;

    extern __shared__ float cumsum[];

    // Single thread computes cumulative sum sequentially (T is small, e.g., 64)
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < T; ++i) {
            sum += x_elem[i];
            cumsum[i] = sum;
        }
    }
    __syncthreads();

    int i = threadIdx.x;
    if (i < T) {
        float cumsum_i = cumsum[i];
        for (int j = 0; j <= i; ++j) {
            out_elem[i * T + j] = expf(cumsum_i - cumsum[j]);
        }
        for (int j = i + 1; j < T; ++j) {
            out_elem[i * T + j] = 0.0f;
        }
    }
}

torch::Tensor segsum_exp_cuda(torch::Tensor x) {
    // x shape: (..., T)
    auto T = x.size(-1);
    auto x_2d = x.reshape({-1, T});
    int total_elements = x_2d.size(0);

    auto out = torch::zeros({total_elements, T, T}, x.options());

    int block_size = T;
    int grid_size = total_elements;
    size_t shared_mem_size = T * sizeof(float);

    segsum_exp_kernel<<<grid_size, block_size, shared_mem_size>>>(
        x_2d.data_ptr<float>(), out.data_ptr<float>(), T, total_elements
    );

    // Reshape output to original leading dims + (T, T)
    auto leading_dims = x.sizes().vec();
    leading_dims.pop_back(); // remove T
    leading_dims.push_back(T);
    leading_dims.push_back(T);
    out = out.reshape(leading_dims);

    return out;
}
"""

segsum_exp_cpp_source = "torch::Tensor segsum_exp_cuda(torch::Tensor x);"

# Compile the inline CUDA code
segsum_exp = load_inline(
    name="segsum_exp",
    cpp_sources=segsum_exp_cpp_source,
    cuda_sources=segsum_exp_source,
    functions=["segsum_exp_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        super(ModelNew, self).__init__()
        assert seq_length % block_len == 0, "Sequence length must be divisible by block length"
        
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.block_len = block_len
        
        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads))
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        
        self.segsum_exp = segsum_exp

    def forward(self, X, initial_states=None):
        # Rearrange into blocks/chunks
        X_blocks, A_blocks, B_blocks, C_blocks = [
            rearrange(x, "b (c l) ... -> b c l ...", l=self.block_len)
            for x in (X, self.A, self.B, self.C)
        ]
        
        A_blocks = rearrange(A_blocks, "b c l h -> b h c l")
        A_cumsum = torch.cumsum(A_blocks, dim=-1)
        
        # 1. Compute diagonal block outputs using custom segsum_exp
        L = self.segsum_exp.segsum_exp_cuda(A_blocks)
        Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", 
                             C_blocks, B_blocks, L, X_blocks)
        
        # 2. Compute intra-chunk states
        decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
        states = torch.einsum("bclhn,bhcl,bclhp->bchpn", 
                            B_blocks, decay_states, X_blocks)
        
        # 3. Compute inter-chunk recurrence
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)
        
        # Use custom segsum_exp for decay_chunk
        padded = F.pad(A_cumsum[:, :, :, -1], (1, 0))
        decay_chunk = self.segsum_exp.segsum_exp_cuda(padded)
        new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
        states = new_states[:, :-1]
        
        # 4. Compute state-to-output conversion
        state_decay_out = torch.exp(A_cumsum)
        Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', 
                           C_blocks, states, state_decay_out)
        
        # Combine diagonal and off-diagonal terms
        Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
        
        return Y