import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for the fused RNN cell kernel
rnn_cell_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void rnn_cell_fused_kernel(
    const float* __restrict__ x,
    const float* __restrict__ h_prev,
    const float* __restrict__ W_ih,
    const float* __restrict__ W_hh,
    const float* __restrict__ b_ih,
    const float* __restrict__ W_ho,
    const float* __restrict__ b_ho,
    float* __restrict__ h_new,
    float* __restrict__ output,
    int batch_size,
    int input_size,
    int hidden_size,
    int output_size
) {
    int b = blockIdx.x;
    int tid = threadIdx.x;

    // Compute new hidden state
    if (tid < hidden_size) {
        float sum = b_ih[tid];
        // Input contribution
        for (int i = 0; i < input_size; ++i) {
            sum += x[b * input_size + i] * W_ih[tid * input_size + i];
        }
        // Hidden contribution
        for (int j = 0; j < hidden_size; ++j) {
            sum += h_prev[b * hidden_size + j] * W_hh[tid * hidden_size + j];
        }
        float val = tanhf(sum);
        h_new[b * hidden_size + tid] = val;
        // Store in shared memory for output computation
        __shared__ float shared_h[256];  // max hidden_size supported
        shared_h[tid] = val;
    }

    __syncthreads();

    // Compute output
    if (tid < output_size) {
        float sum = b_ho[tid];
        for (int j = 0; j < hidden_size; ++j) {
            sum += shared_h[j] * W_ho[tid * hidden_size + j];
        }
        output[b * output_size + tid] = sum;
    }
}

torch::Tensor rnn_cell_fused_cuda(
    torch::Tensor x,
    torch::Tensor h_prev,
    torch::Tensor W_ih,
    torch::Tensor W_hh,
    torch::Tensor b_ih,
    torch::Tensor W_ho,
    torch::Tensor b_ho
) {
    int batch_size = x.size(0);
    int input_size = x.size(1);
    int hidden_size = h_prev.size(1);
    int output_size = W_ho.size(0);

    auto h_new = torch::empty({batch_size, hidden_size}, x.options());
    auto output = torch::empty({batch_size, output_size}, x.options());

    const int block_size = hidden_size;  // must be >= hidden_size
    const int num_blocks = batch_size;

    rnn_cell_fused_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        h_prev.data_ptr<float>(),
        W_ih.data_ptr<float>(),
        W_hh.data_ptr<float>(),
        b_ih.data_ptr<float>(),
        W_ho.data_ptr<float>(),
        b_ho.data_ptr<float>(),
        h_new.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input_size,
        hidden_size,
        output_size
    );

    return torch::stack({h_new, output}, 0);  // return both tensors as a tuple-like stack
}
"""

rnn_cell_cpp_source = """
torch::Tensor rnn_cell_fused_cuda(
    torch::Tensor x,
    torch::Tensor h_prev,
    torch::Tensor W_ih,
    torch::Tensor W_hh,
    torch::Tensor b_ih,
    torch::Tensor W_ho,
    torch::Tensor b_ho
);
"""

# Compile the inline CUDA code
rnn_cell_fused = load_inline(
    name="rnn_cell_fused",
    cpp_sources=rnn_cell_cpp_source,
    cuda_sources=rnn_cell_source,
    functions=["rnn_cell_fused_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Original i2h weight: (hidden_size, input_size + hidden_size)
        i2h = nn.Linear(input_size + hidden_size, hidden_size)
        # Split into W_ih and W_hh
        self.W_ih = i2h.weight[:, :input_size].detach().clone()  # (hidden_size, input_size)
        self.W_hh = i2h.weight[:, input_size:].detach().clone()  # (hidden_size, hidden_size)
        self.b_ih = i2h.bias.detach().clone()                    # (hidden_size,)

        # h2o layer
        h2o = nn.Linear(hidden_size, output_size)
        self.W_ho = h2o.weight.detach().clone()                  # (output_size, hidden_size)
        self.b_ho = h2o.bias.detach().clone()                    # (output_size,)

        # Register as buffers so they are moved to the correct device
        self.register_buffer('W_ih_buf', self.W_ih)
        self.register_buffer('W_hh_buf', self.W_hh)
        self.register_buffer('b_ih_buf', self.b_ih)
        self.register_buffer('W_ho_buf', self.W_ho)
        self.register_buffer('b_ho_buf', self.b_ho)

        self.rnn_cell = rnn_cell_fused

    def forward(self, x: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        seq_len, batch_size, _ = x.size()
        hidden = h0.to(x.device)
        outputs = []

        for t in range(seq_len):
            # Call fused kernel: returns a tensor of shape (2, batch_size, ...)
            # where [0] is new hidden, [1] is output
            result = self.rnn_cell.rnn_cell_fused_cuda(
                x[t].contiguous(),
                hidden.contiguous(),
                self.W_ih_buf,
                self.W_hh_buf,
                self.b_ih_buf,
                self.W_ho_buf,
                self.b_ho_buf
            )
            hidden = result[0]
            outputs.append(result[1])

        return torch.stack(outputs, dim=0)