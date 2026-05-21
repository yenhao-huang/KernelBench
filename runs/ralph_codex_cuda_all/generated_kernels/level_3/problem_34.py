import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

rnn_cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void rnn_hidden_kernel(
    const float* __restrict__ x_t,
    const float* __restrict__ h_prev,
    const float* __restrict__ w_ih,
    const float* __restrict__ b_ih,
    float* __restrict__ h_cur,
    int batch_size,
    int input_size,
    int hidden_size
) {
    int bh = blockIdx.x;
    int b = bh / hidden_size;
    int h = bh - b * hidden_size;
    int tid = threadIdx.x;
    int total = input_size + hidden_size;

    float sum = 0.0f;
    const float* wrow = w_ih + h * total;

    for (int k = tid; k < input_size; k += blockDim.x) {
        sum += x_t[b * input_size + k] * wrow[k];
    }
    for (int k = tid; k < hidden_size; k += blockDim.x) {
        sum += h_prev[b * hidden_size + k] * wrow[input_size + k];
    }

    __shared__ float sh[256];
    sh[tid] = sum;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        h_cur[b * hidden_size + h] = tanhf(sh[0] + b_ih[h]);
    }
}

__global__ void rnn_output_kernel(
    const float* __restrict__ h_cur,
    const float* __restrict__ w_ho,
    const float* __restrict__ b_ho,
    float* __restrict__ out_t,
    int batch_size,
    int hidden_size,
    int output_size
) {
    int bo = blockIdx.x;
    int b = bo / output_size;
    int o = bo - b * output_size;
    int tid = threadIdx.x;

    float sum = 0.0f;
    const float* wrow = w_ho + o * hidden_size;

    for (int k = tid; k < hidden_size; k += blockDim.x) {
        sum += h_cur[b * hidden_size + k] * wrow[k];
    }

    __shared__ float sh[256];
    sh[tid] = sum;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        out_t[b * output_size + o] = sh[0] + b_ho[o];
    }
}

torch::Tensor rnn_forward_cuda(
    torch::Tensor x,
    torch::Tensor h0,
    torch::Tensor w_ih,
    torch::Tensor b_ih,
    torch::Tensor w_ho,
    torch::Tensor b_ho
) {
    int seq_len = (int)x.size(0);
    int batch_size = (int)x.size(1);
    int input_size = (int)x.size(2);
    int hidden_size = (int)h0.size(1);
    int output_size = (int)b_ho.size(0);

    auto h_a = torch::empty_like(h0);
    auto h_b = torch::empty_like(h0);
    auto out = torch::empty({seq_len, batch_size, output_size}, x.options());

    const float* prev_ptr = h0.data_ptr<float>();
    float* cur_ptr = h_a.data_ptr<float>();

    const int threads = 256;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    for (int t = 0; t < seq_len; ++t) {
        const float* x_t = x.data_ptr<float>() + (long long)t * batch_size * input_size;
        float* out_t = out.data_ptr<float>() + (long long)t * batch_size * output_size;

        rnn_hidden_kernel<<<batch_size * hidden_size, threads, 0, stream>>>(
            x_t, prev_ptr, w_ih.data_ptr<float>(), b_ih.data_ptr<float>(),
            cur_ptr, batch_size, input_size, hidden_size
        );

        rnn_output_kernel<<<batch_size * output_size, threads, 0, stream>>>(
            cur_ptr, w_ho.data_ptr<float>(), b_ho.data_ptr<float>(),
            out_t, batch_size, hidden_size, output_size
        );

        prev_ptr = cur_ptr;
        cur_ptr = (cur_ptr == h_a.data_ptr<float>()) ? h_b.data_ptr<float>() : h_a.data_ptr<float>();
    }

    return out;
}
"""

rnn_cpp_source = r"""
torch::Tensor rnn_forward_cuda(
    torch::Tensor x,
    torch::Tensor h0,
    torch::Tensor w_ih,
    torch::Tensor b_ih,
    torch::Tensor w_ho,
    torch::Tensor b_ho
);
"""

rnn_ext = load_inline(
    name="vanilla_rnn_fused_cuda_fp32",
    cpp_sources=rnn_cpp_source,
    cuda_sources=rnn_cuda_source,
    functions=["rnn_forward_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        return rnn_ext.rnn_forward_cuda(
            x,
            h0,
            self.i2h.weight,
            self.i2h.bias,
            self.h2o.weight,
            self.h2o.bias,
        )