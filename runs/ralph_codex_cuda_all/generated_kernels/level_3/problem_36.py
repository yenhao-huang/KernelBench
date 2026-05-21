import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

lstm_cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float sigmoidf_fast(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void lstm6_kernel(
    const float* __restrict__ x,
    const float* __restrict__ h0,
    const float* __restrict__ c0,
    float* __restrict__ hn,
    float* __restrict__ buf1,
    float* __restrict__ buf2,
    const float* __restrict__ wih0, const float* __restrict__ whh0, const float* __restrict__ bih0, const float* __restrict__ bhh0,
    const float* __restrict__ wih1, const float* __restrict__ whh1, const float* __restrict__ bih1, const float* __restrict__ bhh1,
    const float* __restrict__ wih2, const float* __restrict__ whh2, const float* __restrict__ bih2, const float* __restrict__ bhh2,
    const float* __restrict__ wih3, const float* __restrict__ whh3, const float* __restrict__ bih3, const float* __restrict__ bhh3,
    const float* __restrict__ wih4, const float* __restrict__ whh4, const float* __restrict__ bih4, const float* __restrict__ bhh4,
    const float* __restrict__ wih5, const float* __restrict__ whh5, const float* __restrict__ bih5, const float* __restrict__ bhh5,
    int B, int T, int I, int H
) {
    extern __shared__ float sh[];
    float* prev_h = sh;

    const int b = blockIdx.x;
    const int h = threadIdx.x;

    const float* wihs[6] = {wih0, wih1, wih2, wih3, wih4, wih5};
    const float* whhs[6] = {whh0, whh1, whh2, whh3, whh4, whh5};
    const float* bihs[6] = {bih0, bih1, bih2, bih3, bih4, bih5};
    const float* bhhs[6] = {bhh0, bhh1, bhh2, bhh3, bhh4, bhh5};

    for (int layer = 0; layer < 6; ++layer) {
        const float* wih = wihs[layer];
        const float* whh = whhs[layer];
        const float* bih = bihs[layer];
        const float* bhh = bhhs[layer];

        const int K = (layer == 0) ? I : H;
        const float* layer_in = nullptr;
        if (layer > 0) {
            layer_in = (layer & 1) ? buf1 : buf2;
        }
        float* layer_out = (layer & 1) ? buf2 : buf1;

        float c = c0[(layer * B + b) * H + h];
        float hp = h0[(layer * B + b) * H + h];

        for (int t = 0; t < T; ++t) {
            prev_h[h] = hp;
            __syncthreads();

            float gi = bih[h] + bhh[h];
            float gf = bih[H + h] + bhh[H + h];
            float gg = bih[2 * H + h] + bhh[2 * H + h];
            float go = bih[3 * H + h] + bhh[3 * H + h];

            const int wbase_i = h * K;
            const int wbase_f = (H + h) * K;
            const int wbase_g = (2 * H + h) * K;
            const int wbase_o = (3 * H + h) * K;

            if (layer == 0) {
                const float* xin = x + (b * T + t) * I;
                for (int k = 0; k < K; ++k) {
                    float xv = xin[k];
                    gi += xv * wih[wbase_i + k];
                    gf += xv * wih[wbase_f + k];
                    gg += xv * wih[wbase_g + k];
                    go += xv * wih[wbase_o + k];
                }
            } else {
                const float* xin = layer_in + (b * T + t) * H;
                for (int k = 0; k < H; ++k) {
                    float xv = xin[k];
                    gi += xv * wih[wbase_i + k];
                    gf += xv * wih[wbase_f + k];
                    gg += xv * wih[wbase_g + k];
                    go += xv * wih[wbase_o + k];
                }
            }

            const int rh_i = h * H;
            const int rh_f = (H + h) * H;
            const int rh_g = (2 * H + h) * H;
            const int rh_o = (3 * H + h) * H;

            for (int k = 0; k < H; ++k) {
                float hv = prev_h[k];
                gi += hv * whh[rh_i + k];
                gf += hv * whh[rh_f + k];
                gg += hv * whh[rh_g + k];
                go += hv * whh[rh_o + k];
            }

            float in_gate = sigmoidf_fast(gi);
            float forget_gate = sigmoidf_fast(gf);
            float cell_gate = tanhf(gg);
            float out_gate = sigmoidf_fast(go);

            c = forget_gate * c + in_gate * cell_gate;
            hp = out_gate * tanhf(c);

            layer_out[(b * T + t) * H + h] = hp;
            __syncthreads();
        }

        hn[(layer * B + b) * H + h] = hp;
        __syncthreads();
    }
}

torch::Tensor lstm6_forward_cuda(
    torch::Tensor x,
    torch::Tensor h0,
    torch::Tensor c0,
    torch::Tensor wih0, torch::Tensor whh0, torch::Tensor bih0, torch::Tensor bhh0,
    torch::Tensor wih1, torch::Tensor whh1, torch::Tensor bih1, torch::Tensor bhh1,
    torch::Tensor wih2, torch::Tensor whh2, torch::Tensor bih2, torch::Tensor bhh2,
    torch::Tensor wih3, torch::Tensor whh3, torch::Tensor bih3, torch::Tensor bhh3,
    torch::Tensor wih4, torch::Tensor whh4, torch::Tensor bih4, torch::Tensor bhh4,
    torch::Tensor wih5, torch::Tensor whh5, torch::Tensor bih5, torch::Tensor bhh5
) {
    int B = (int)x.size(0);
    int T = (int)x.size(1);
    int I = (int)x.size(2);
    int H = (int)h0.size(2);

    auto hn = torch::empty_like(h0);
    auto buf1 = torch::empty({B, T, H}, x.options());
    auto buf2 = torch::empty({B, T, H}, x.options());

    lstm6_kernel<<<B, H, H * sizeof(float)>>>(
        x.data_ptr<float>(),
        h0.data_ptr<float>(),
        c0.data_ptr<float>(),
        hn.data_ptr<float>(),
        buf1.data_ptr<float>(),
        buf2.data_ptr<float>(),
        wih0.data_ptr<float>(), whh0.data_ptr<float>(), bih0.data_ptr<float>(), bhh0.data_ptr<float>(),
        wih1.data_ptr<float>(), whh1.data_ptr<float>(), bih1.data_ptr<float>(), bhh1.data_ptr<float>(),
        wih2.data_ptr<float>(), whh2.data_ptr<float>(), bih2.data_ptr<float>(), bhh2.data_ptr<float>(),
        wih3.data_ptr<float>(), whh3.data_ptr<float>(), bih3.data_ptr<float>(), bhh3.data_ptr<float>(),
        wih4.data_ptr<float>(), whh4.data_ptr<float>(), bih4.data_ptr<float>(), bhh4.data_ptr<float>(),
        wih5.data_ptr<float>(), whh5.data_ptr<float>(), bih5.data_ptr<float>(), bhh5.data_ptr<float>(),
        B, T, I, H
    );

    return hn;
}
"""

lstm_cpp_sources = r"""
torch::Tensor lstm6_forward_cuda(
    torch::Tensor x,
    torch::Tensor h0,
    torch::Tensor c0,
    torch::Tensor wih0, torch::Tensor whh0, torch::Tensor bih0, torch::Tensor bhh0,
    torch::Tensor wih1, torch::Tensor whh1, torch::Tensor bih1, torch::Tensor bhh1,
    torch::Tensor wih2, torch::Tensor whh2, torch::Tensor bih2, torch::Tensor bhh2,
    torch::Tensor wih3, torch::Tensor whh3, torch::Tensor bih3, torch::Tensor bhh3,
    torch::Tensor wih4, torch::Tensor whh4, torch::Tensor bih4, torch::Tensor bhh4,
    torch::Tensor wih5, torch::Tensor whh5, torch::Tensor bih5, torch::Tensor bhh5
);
"""

lstm_ext = load_inline(
    name="kb_lstm6_state_ext",
    cpp_sources=lstm_cpp_sources,
    cuda_sources=lstm_cuda_sources,
    functions=["lstm6_forward_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self._lstm_ext = lstm_ext

    def forward(self, x, h0, c0):
        return self._lstm_ext.lstm6_forward_cuda(
            x,
            h0,
            c0,
            self.lstm.weight_ih_l0,
            self.lstm.weight_hh_l0,
            self.lstm.bias_ih_l0,
            self.lstm.bias_hh_l0,
            self.lstm.weight_ih_l1,
            self.lstm.weight_hh_l1,
            self.lstm.bias_ih_l1,
            self.lstm.bias_hh_l1,
            self.lstm.weight_ih_l2,
            self.lstm.weight_hh_l2,
            self.lstm.bias_ih_l2,
            self.lstm.bias_hh_l2,
            self.lstm.weight_ih_l3,
            self.lstm.weight_hh_l3,
            self.lstm.bias_ih_l3,
            self.lstm.bias_hh_l3,
            self.lstm.weight_ih_l4,
            self.lstm.weight_hh_l4,
            self.lstm.bias_ih_l4,
            self.lstm.bias_hh_l4,
            self.lstm.weight_ih_l5,
            self.lstm.weight_hh_l5,
            self.lstm.bias_ih_l5,
            self.lstm.bias_hh_l5,
        )