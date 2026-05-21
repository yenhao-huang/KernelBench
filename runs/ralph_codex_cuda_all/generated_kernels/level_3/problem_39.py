import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

gru_cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

__device__ __forceinline__ float sigmoidf_fast(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void copy_hidden_kernel(const float* __restrict__ h0, float* __restrict__ h,
                                   int layer, int B, int H) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = B * H;
    if (idx < n) h[idx] = h0[layer * n + idx];
}

__global__ void store_hidden_kernel(const float* __restrict__ h, float* __restrict__ hn,
                                    int layer, int B, int H) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = B * H;
    if (idx < n) hn[layer * n + idx] = h[idx];
}

__global__ void gru_step_kernel(const float* __restrict__ x,
                                const float* __restrict__ hprev,
                                float* __restrict__ hnext,
                                const float* __restrict__ w_ih,
                                const float* __restrict__ w_hh,
                                const float* __restrict__ b_ih,
                                const float* __restrict__ b_hh,
                                int B, int I, int H) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H;
    if (idx >= total) return;

    int b = idx / H;
    int j = idx - b * H;

    const float* xb = x + b * I;
    const float* hb = hprev + b * H;

    float ir = b_ih[j];
    float iz = b_ih[H + j];
    float in = b_ih[2 * H + j];

    for (int k = 0; k < I; ++k) {
        float xv = xb[k];
        ir += xv * w_ih[j * I + k];
        iz += xv * w_ih[(H + j) * I + k];
        in += xv * w_ih[(2 * H + j) * I + k];
    }

    float hr = b_hh[j];
    float hz = b_hh[H + j];
    float hn = b_hh[2 * H + j];

    for (int k = 0; k < H; ++k) {
        float hv = hb[k];
        hr += hv * w_hh[j * H + k];
        hz += hv * w_hh[(H + j) * H + k];
        hn += hv * w_hh[(2 * H + j) * H + k];
    }

    float r = sigmoidf_fast(ir + hr);
    float z = sigmoidf_fast(iz + hz);
    float n = tanhf(in + r * hn);
    float oldh = hb[j];
    hnext[idx] = (1.0f - z) * n + z * oldh;
}

torch::Tensor gru_forward_cuda(torch::Tensor x,
                               torch::Tensor h0,
                               std::vector<torch::Tensor> w_ih,
                               std::vector<torch::Tensor> w_hh,
                               std::vector<torch::Tensor> b_ih,
                               std::vector<torch::Tensor> b_hh,
                               bool batch_first) {
    int64_t S = batch_first ? x.size(1) : x.size(0);
    int64_t B = batch_first ? x.size(0) : x.size(1);
    int64_t I0 = x.size(2);
    int64_t L = h0.size(0);
    int64_t H = h0.size(2);

    auto opts = x.options();
    auto layer_in = x.contiguous();
    torch::Tensor layer_out;
    auto hcur = torch::empty({B, H}, opts);
    auto hnext = torch::empty({B, H}, opts);
    auto hn = torch::empty_like(h0);

    const int threads = 256;
    const int hidden_blocks = (B * H + threads - 1) / threads;

    for (int l = 0; l < L; ++l) {
        int I = (l == 0) ? (int)I0 : (int)H;
        layer_out = torch::empty({S, B, H}, opts);

        copy_hidden_kernel<<<hidden_blocks, threads>>>(
            h0.data_ptr<float>(), hcur.data_ptr<float>(), l, (int)B, (int)H
        );

        for (int t = 0; t < S; ++t) {
            const float* xt;
            if (batch_first && l == 0) {
                xt = layer_in.data_ptr<float>() + t * I;
            } else {
                xt = layer_in.data_ptr<float>() + t * B * I;
            }

            float* out_t = layer_out.data_ptr<float>() + t * B * H;

            gru_step_kernel<<<hidden_blocks, threads>>>(
                xt, hcur.data_ptr<float>(), hnext.data_ptr<float>(),
                w_ih[l].data_ptr<float>(), w_hh[l].data_ptr<float>(),
                b_ih[l].data_ptr<float>(), b_hh[l].data_ptr<float>(),
                (int)B, I, (int)H
            );

            cudaMemcpyAsync(out_t, hnext.data_ptr<float>(), B * H * sizeof(float), cudaMemcpyDeviceToDevice);
            auto tmp = hcur;
            hcur = hnext;
            hnext = tmp;
        }

        store_hidden_kernel<<<hidden_blocks, threads>>>(
            hcur.data_ptr<float>(), hn.data_ptr<float>(), l, (int)B, (int)H
        );

        layer_in = layer_out;
    }

    if (batch_first) {
        return layer_out.permute({1, 0, 2}).contiguous();
    }
    return layer_out;
}
"""

gru_cpp_source = r"""
#include <torch/extension.h>
#include <vector>

torch::Tensor gru_forward_cuda(torch::Tensor x,
                               torch::Tensor h0,
                               std::vector<torch::Tensor> w_ih,
                               std::vector<torch::Tensor> w_hh,
                               std::vector<torch::Tensor> b_ih,
                               std::vector<torch::Tensor> b_hh,
                               bool batch_first);
"""

gru_ext = load_inline(
    name="custom_gru_forward_ext",
    cpp_sources=gru_cpp_source,
    cuda_sources=gru_cuda_source,
    functions=["gru_forward_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bias, batch_first, dropout=0, bidirectional=False)
        self.num_layers = num_layers
        self.batch_first = batch_first

    def forward(self, x, h0):
        w_ih = []
        w_hh = []
        b_ih = []
        b_hh = []
        for i in range(self.num_layers):
            w_ih.append(getattr(self.gru, f"weight_ih_l{i}").contiguous())
            w_hh.append(getattr(self.gru, f"weight_hh_l{i}").contiguous())
            b_ih.append(getattr(self.gru, f"bias_ih_l{i}").contiguous())
            b_hh.append(getattr(self.gru, f"bias_hh_l{i}").contiguous())
        return gru_ext.gru_forward_cuda(x, h0, w_ih, w_hh, b_ih, b_hh, self.batch_first)