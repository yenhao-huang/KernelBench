import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

gru_cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

__device__ __forceinline__ float sigmoidf_fast(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void copy_hidden_kernel(const float* __restrict__ h0, float* __restrict__ h, int B, int H) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H;
    if (idx < total) h[idx] = h0[idx];
}

__global__ void gru_step_kernel(
    const float* __restrict__ x,
    const float* __restrict__ hprev,
    const float* __restrict__ w_ih,
    const float* __restrict__ w_hh,
    const float* __restrict__ b_ih,
    const float* __restrict__ b_hh,
    float* __restrict__ hnext,
    float* __restrict__ out,
    int B,
    int K,
    int H
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H;
    if (idx >= total) return;

    int b = idx / H;
    int h = idx - b * H;

    float ir = b_ih[h];
    float iz = b_ih[H + h];
    float in = b_ih[2 * H + h];

    const float* xb = x + b * K;
    for (int k = 0; k < K; ++k) {
        float xv = xb[k];
        ir += xv * w_ih[h * K + k];
        iz += xv * w_ih[(H + h) * K + k];
        in += xv * w_ih[(2 * H + h) * K + k];
    }

    float hr = b_hh[h];
    float hz = b_hh[H + h];
    float hn = b_hh[2 * H + h];

    const float* hp = hprev + b * H;
    for (int j = 0; j < H; ++j) {
        float hv = hp[j];
        hr += hv * w_hh[h * H + j];
        hz += hv * w_hh[(H + h) * H + j];
        hn += hv * w_hh[(2 * H + h) * H + j];
    }

    float r = sigmoidf_fast(ir + hr);
    float z = sigmoidf_fast(iz + hz);
    float n = tanhf(in + r * hn);
    float oldh = hprev[b * H + h];
    float nh = (1.0f - z) * n + z * oldh;

    hnext[b * H + h] = nh;
    out[b * H + h] = nh;
}

torch::Tensor gru_forward_cuda(
    torch::Tensor x,
    torch::Tensor h0,
    std::vector<torch::Tensor> weights,
    bool batch_first
) {
    int64_t S = batch_first ? x.size(1) : x.size(0);
    int64_t B = batch_first ? x.size(0) : x.size(1);
    int64_t input_size = x.size(2);
    int64_t L = h0.size(0);
    int64_t H = h0.size(2);

    auto opts = x.options();
    auto hn = torch::empty_like(h0);

    torch::Tensor prev_out;
    torch::Tensor curr_out = torch::empty({S, B, H}, opts);
    torch::Tensor h_a = torch::empty({B, H}, opts);
    torch::Tensor h_b = torch::empty({B, H}, opts);

    const int threads = 256;
    const int blocks_h = (B * H + threads - 1) / threads;

    for (int64_t l = 0; l < L; ++l) {
        const float* h0_l = h0.data_ptr<float>() + l * B * H;
        copy_hidden_kernel<<<blocks_h, threads>>>(h0_l, h_a.data_ptr<float>(), B, H);

        auto w_ih = weights[l * 4 + 0].contiguous();
        auto w_hh = weights[l * 4 + 1].contiguous();
        auto b_ih = weights[l * 4 + 2].contiguous();
        auto b_hh = weights[l * 4 + 3].contiguous();

        int64_t K = (l == 0) ? input_size : H;

        for (int64_t t = 0; t < S; ++t) {
            const float* x_t;
            if (l == 0) {
                if (batch_first) {
                    x_t = x.data_ptr<float>() + t * input_size;
                } else {
                    x_t = x.data_ptr<float>() + t * B * input_size;
                }
            } else {
                x_t = prev_out.data_ptr<float>() + t * B * H;
            }

            float* out_t = curr_out.data_ptr<float>() + t * B * H;

            gru_step_kernel<<<blocks_h, threads>>>(
                x_t,
                h_a.data_ptr<float>(),
                w_ih.data_ptr<float>(),
                w_hh.data_ptr<float>(),
                b_ih.data_ptr<float>(),
                b_hh.data_ptr<float>(),
                h_b.data_ptr<float>(),
                out_t,
                B,
                K,
                H
            );

            auto tmp = h_a;
            h_a = h_b;
            h_b = tmp;
        }

        copy_hidden_kernel<<<blocks_h, threads>>>(
            h_a.data_ptr<float>(),
            hn.data_ptr<float>() + l * B * H,
            B,
            H
        );

        prev_out = curr_out;
        if (l + 1 < L) {
            curr_out = torch::empty({S, B, H}, opts);
        }
    }

    return hn;
}
"""

gru_cpp_src = r"""
torch::Tensor gru_forward_cuda(
    torch::Tensor x,
    torch::Tensor h0,
    std::vector<torch::Tensor> weights,
    bool batch_first
);
"""

gru_ext = load_inline(
    name="custom_gru_forward_ext",
    cpp_sources=gru_cpp_src,
    cuda_sources=gru_cuda_src,
    functions=["gru_forward_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bias, batch_first, dropout=0, bidirectional=False)
        self.batch_first = batch_first
        self.num_layers = num_layers

    def forward(self, x, h0):
        weights = []
        for i in range(self.num_layers):
            weights.append(getattr(self.gru, "weight_ih_l%d" % i))
            weights.append(getattr(self.gru, "weight_hh_l%d" % i))
            weights.append(getattr(self.gru, "bias_ih_l%d" % i))
            weights.append(getattr(self.gru, "bias_hh_l%d" % i))
        return gru_ext.gru_forward_cuda(x, h0, weights, self.batch_first)