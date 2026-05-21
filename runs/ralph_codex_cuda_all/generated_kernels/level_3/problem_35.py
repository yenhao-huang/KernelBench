import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

lstm_cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

__device__ __forceinline__ float sigmoidf_fast(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void lstm_step_kernel(
    const float* __restrict__ inp,
    const float* __restrict__ h_prev,
    const float* __restrict__ c_prev,
    const float* __restrict__ w_ih,
    const float* __restrict__ w_hh,
    const float* __restrict__ b_ih,
    const float* __restrict__ b_hh,
    float* __restrict__ h_next,
    float* __restrict__ c_next,
    int B, int I, int H
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H;
    if (idx >= total) return;

    int h = idx % H;
    int b = idx / H;

    float gates[4];

    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        float acc = b_ih[g * H + h] + b_hh[g * H + h];

        const float* wi = w_ih + (g * H + h) * I;
        const float* wh = w_hh + (g * H + h) * H;
        const float* xi = inp + b * I;
        const float* hp = h_prev + b * H;

        for (int k = 0; k < I; ++k) acc += xi[k] * wi[k];
        for (int k = 0; k < H; ++k) acc += hp[k] * wh[k];

        gates[g] = acc;
    }

    float i_gate = sigmoidf_fast(gates[0]);
    float f_gate = sigmoidf_fast(gates[1]);
    float g_gate = tanhf(gates[2]);
    float o_gate = sigmoidf_fast(gates[3]);

    float c = f_gate * c_prev[idx] + i_gate * g_gate;
    float hn = o_gate * tanhf(c);

    c_next[idx] = c;
    h_next[idx] = hn;
}

__global__ void copy_last_to_input_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int B, int H
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * H) dst[idx] = src[idx];
}

__global__ void linear_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int B, int H, int O
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * O;
    if (idx >= total) return;

    int o = idx % O;
    int b = idx / O;

    float acc = bias[o];
    const float* xb = x + b * H;
    const float* wo = w + o * H;

    for (int k = 0; k < H; ++k) acc += xb[k] * wo[k];
    out[idx] = acc;
}

torch::Tensor lstm_forward_cuda(
    torch::Tensor x,
    torch::Tensor h0,
    torch::Tensor c0,
    std::vector<torch::Tensor> w_ih,
    std::vector<torch::Tensor> w_hh,
    std::vector<torch::Tensor> b_ih,
    std::vector<torch::Tensor> b_hh,
    torch::Tensor fc_w,
    torch::Tensor fc_b
) {
    int B = x.size(0);
    int T = x.size(1);
    int input_size = x.size(2);
    int L = (int)w_ih.size();
    int H = h0.size(2);
    int O = fc_w.size(0);

    auto opts = x.options();
    auto h_buf_a = torch::empty({B, H}, opts);
    auto h_buf_b = torch::empty({B, H}, opts);
    auto c_buf_a = torch::empty({B, H}, opts);
    auto c_buf_b = torch::empty({B, H}, opts);

    auto layer_out = torch::empty({B, T, H}, opts);
    auto final_h = torch::empty({B, H}, opts);

    const int threads = 128;
    const int bh_blocks = (B * H + threads - 1) / threads;

    const float* layer_input = x.data_ptr<float>();
    int cur_I = input_size;

    for (int layer = 0; layer < L; ++layer) {
        cudaMemcpy(
            h_buf_a.data_ptr<float>(),
            h0.data_ptr<float>() + layer * B * H,
            B * H * sizeof(float),
            cudaMemcpyDeviceToDevice
        );
        cudaMemcpy(
            c_buf_a.data_ptr<float>(),
            c0.data_ptr<float>() + layer * B * H,
            B * H * sizeof(float),
            cudaMemcpyDeviceToDevice
        );

        for (int t = 0; t < T; ++t) {
            const float* xt = layer_input + ((long long)t * cur_I);
            if (layer == 0) {
                xt = x.data_ptr<float>() + ((long long)t * input_size);
            } else {
                xt = layer_input + ((long long)t * B * cur_I);
            }

            lstm_step_kernel<<<bh_blocks, threads>>>(
                xt,
                h_buf_a.data_ptr<float>(),
                c_buf_a.data_ptr<float>(),
                w_ih[layer].data_ptr<float>(),
                w_hh[layer].data_ptr<float>(),
                b_ih[layer].data_ptr<float>(),
                b_hh[layer].data_ptr<float>(),
                h_buf_b.data_ptr<float>(),
                c_buf_b.data_ptr<float>(),
                B, cur_I, H
            );

            float* out_t = layer_out.data_ptr<float>() + (long long)t * B * H;
            copy_last_to_input_kernel<<<bh_blocks, threads>>>(
                h_buf_b.data_ptr<float>(), out_t, B, H
            );

            std::swap(h_buf_a, h_buf_b);
            std::swap(c_buf_a, c_buf_b);
        }

        layer_input = layer_out.data_ptr<float>();
        cur_I = H;
    }

    copy_last_to_input_kernel<<<bh_blocks, threads>>>(
        h_buf_a.data_ptr<float>(), final_h.data_ptr<float>(), B, H
    );

    auto out = torch::empty({B, O}, opts);
    int lin_blocks = (B * O + threads - 1) / threads;
    linear_kernel<<<lin_blocks, threads>>>(
        final_h.data_ptr<float>(),
        fc_w.data_ptr<float>(),
        fc_b.data_ptr<float>(),
        out.data_ptr<float>(),
        B, H, O
    );

    return out;
}
"""

lstm_cpp_source = r"""
#include <torch/extension.h>
#include <vector>

torch::Tensor lstm_forward_cuda(
    torch::Tensor x,
    torch::Tensor h0,
    torch::Tensor c0,
    std::vector<torch::Tensor> w_ih,
    std::vector<torch::Tensor> w_hh,
    std::vector<torch::Tensor> b_ih,
    std::vector<torch::Tensor> b_hh,
    torch::Tensor fc_w,
    torch::Tensor fc_b
);
"""

lstm_inline = load_inline(
    name="custom_lstm_kernelbench_fp32",
    cpp_sources=lstm_cpp_source,
    cuda_sources=lstm_cuda_source,
    functions=["lstm_forward_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0=None, c0=None):
        batch_size = x.size(0)

        if h0 is None:
            h0 = torch.randn(self.num_layers, batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        if c0 is None:
            c0 = torch.randn(self.num_layers, batch_size, self.hidden_size, device=x.device, dtype=x.dtype)

        w_ih = []
        w_hh = []
        b_ih = []
        b_hh = []
        for layer in range(self.num_layers):
            w_ih.append(getattr(self.lstm, f"weight_ih_l{layer}").contiguous())
            w_hh.append(getattr(self.lstm, f"weight_hh_l{layer}").contiguous())
            b_ih.append(getattr(self.lstm, f"bias_ih_l{layer}").contiguous())
            b_hh.append(getattr(self.lstm, f"bias_hh_l{layer}").contiguous())

        return lstm_inline.lstm_forward_cuda(
            x.contiguous(),
            h0.contiguous(),
            c0.contiguous(),
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            self.fc.weight.contiguous(),
            self.fc.bias.contiguous(),
        )