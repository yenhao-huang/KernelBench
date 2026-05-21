import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused segment sum + exponential
segsum_exp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void segsum_exp_kernel(const float* A, float* L, int b, int h, int c, int l) {
    int idx = blockIdx.x;
    int total_blocks = b * h * c;
    if (idx >= total_blocks) return;
    
    int bc = idx % (b * c);
    int b_idx = bc / c;
    int c_idx = bc % c;
    int h_idx = idx / (b * c);
    
    const float* A_block = A + ((b_idx * h + h_idx) * c + c_idx) * l;
    float* L_block = L + (((b_idx * h + h_idx) * c + c_idx) * l * l);
    
    extern __shared__ float shared[];
    float* A_shared = shared;
    float* cumsum = shared + l;
    
    for (int i = threadIdx.x; i < l; i += blockDim.x) {
        A_shared[i] = A_block[i];
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        cumsum[0] = A_shared[0];
        for (int i = 1; i < l; ++i) {
            cumsum[i] = cumsum[i-1] + A_shared[i];
        }
    }
    __syncthreads();
    
    int i = threadIdx.x;
    if (i < l) {
        float cumsum_i_minus_1 = (i > 0) ? cumsum[i-1] : 0.0f;
        for (int j = i; j < l; ++j) {
            float sum = cumsum[j] - cumsum_i_minus_1;
            L_block[i * l + j] = expf(sum);
        }
        for (int j = 0; j < i; ++j) {
            L_block[i * l + j] = 0.0f;
        }
    }
}

torch::Tensor segsum_exp_cuda(torch::Tensor A) {
    auto sizes = A.sizes();
    int b = sizes[0];
    int h = sizes[1];
    int c = sizes[2];
    int l = sizes[3];
    
    auto L = torch::zeros({b, h, c, l, l}, A.options());
    
    int threads = l;
    int blocks = b * h * c;
    size_t shared_mem = 2 * l * sizeof(float);
    
    segsum_exp_kernel<<<blocks, threads, shared_mem>>>(A.data_ptr<float>(), L.data_ptr<float>(), b, h, c, l);
    
    return L;
}
"""

segsum_exp_cpp_source = "torch::Tensor segsum_exp_cuda(torch::Tensor A);"

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
        
        # Initialize parameters
        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads))
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        
        # Custom CUDA operator
        self.segsum_exp = segsum_exp
    
    def forward(self, X, initial_states=None):
        # Rearrange into blocks/chunks
        X_blocks, A_blocks, B_blocks, C_blocks = [
            rearrange(x, "b (c l) ... -> b c l ...", l=self.block_len)
            for x in (X, self.A, self.B, self.C)
        ]
        
        A_blocks = rearrange(A_blocks, "b c l h -> b h c l")
        A_cumsum = torch.cumsum(A_blocks, dim=-1)
        
        # 1. Compute diagonal block outputs using fused segsum+exp
        L = self.segsum_exp.segsum_exp_cuda(A_blocks.contiguous())
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
        
        A_last = A_cumsum[:, :, :, -1]  # (b, h, c)
        A_last_padded = F.pad(A_last, (1, 0))  # (b, h, c+1)
        decay_chunk = self.segsum_exp.segsum_exp_cuda(A_last_padded.contiguous())
        
        new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
        return new_states[:, -1]