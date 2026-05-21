import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

gru_cpp_source = """
torch::Tensor gru_forward_cuda(torch::Tensor x, torch::Tensor h0, torch::TensorList weights, int64_t num_layers, bool batch_first);
"""

gru_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

__device__ __forceinline__ float sigmoidf_fast(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void gru_direction_kernel(
    const float* __restrict__ inp,
    const float* __restrict__ h0,
    const float* __restrict__ w_ih,
    const float* __restrict__ w_hh,
    const float* __restrict__ b_ih,
    const float* __restrict__ b_hh,
    float* __restrict__ out_dir,
    float* __restrict__ hn,
    int seq_len,
    int batch,
    int in_size,
    int hidden,
    int dir,
    int layer_h0_offset
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * hidden;
    if (idx >= total) return;

    int b = idx / hidden;
    int h = idx - b * hidden;

    float prev = h0[layer_h0_offset + b * hidden + h];

    if (dir == 0) {
        for (int t = 0; t < seq_len; ++t) {
            const float* xptr = inp + (t * batch + b) * in_size;

            float ir = b_ih[h];
            float iz = b_ih[hidden + h];
            float in = b_ih[2 * hidden + h];
            for (int k = 0; k < in_size; ++k) {
                float xv = xptr[k];
                ir += xv * w_ih[h * in_size + k];
                iz += xv * w_ih[(hidden + h) * in_size + k];
                in += xv * w_ih[(2 * hidden + h) * in_size + k];
            }

            float hr = b_hh[h];
            float hz = b_hh[hidden + h];
            float hn_lin = b_hh[2 * hidden + h];
            const float* hprev_vec;
            if (t == 0) {
                hprev_vec = h0 + layer_h0_offset + b * hidden;
            } else {
                hprev_vec = out_dir + ((t - 1) * batch + b) * hidden;
            }

            for (int k = 0; k < hidden; ++k) {
                float hv = hprev_vec[k];
                hr += hv * w_hh[h * hidden + k];
                hz += hv * w_hh[(hidden + h) * hidden + k];
                hn_lin += hv * w_hh[(2 * hidden + h) * hidden + k];
            }

            float r = sigmoidf_fast(ir + hr);
            float z = sigmoidf_fast(iz + hz);
            float n = tanhf(in + r * hn_lin);
            prev = (1.0f - z) * n + z * prev;
            out_dir[(t * batch + b) * hidden + h] = prev;
        }
    } else {
        for (int tt = 0; tt < seq_len; ++tt) {
            int t = seq_len - 1 - tt;
            const float* xptr = inp + (t * batch + b) * in_size;

            float ir = b_ih[h];
            float iz = b_ih[hidden + h];
            float in = b_ih[2 * hidden + h];
            for (int k = 0; k < in_size; ++k) {
                float xv = xptr[k];
                ir += xv * w_ih[h * in_size + k];
                iz += xv * w_ih[(hidden + h) * in_size + k];
                in += xv * w_ih[(2 * hidden + h) * in_size + k];
            }

            float hr = b_hh[h];
            float hz = b_hh[hidden + h];
            float hn_lin = b_hh[2 * hidden + h];
            const float* hprev_vec;
            if (tt == 0) {
                hprev_vec = h0 + layer_h0_offset + b * hidden;
            } else {
                hprev_vec = out_dir + ((t + 1) * batch + b) * hidden;
            }

            for (int k = 0; k < hidden; ++k) {
                float hv = hprev_vec[k];
                hr += hv * w_hh[h * hidden + k];
                hz += hv * w_hh[(hidden + h) * hidden + k];
                hn_lin += hv * w_hh[(2 * hidden + h) * hidden + k];
            }

            float r = sigmoidf_fast(ir + hr);
            float z = sigmoidf_fast(iz + hz);
            float n = tanhf(in + r * hn_lin);
            prev = (1.0f - z) * n + z * prev;
            out_dir[(t * batch + b) * hidden + h] = prev;
        }
    }

    hn[layer_h0_offset + b * hidden + h] = prev;
}

__global__ void concat_dirs_kernel(
    const float* __restrict__ fwd,
    const float* __restrict__ rev,
    float* __restrict__ out,
    int total,
    int hidden
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int h = idx % hidden;
    int row = idx / hidden;
    out[row * (2 * hidden) + h] = fwd[idx];
    out[row * (2 * hidden) + hidden + h] = rev[idx];
}

torch::Tensor gru_forward_cuda(torch::Tensor x, torch::Tensor h0, torch::TensorList weights, int64_t num_layers, bool batch_first) {
    auto xc = batch_first ? x.transpose(0, 1).contiguous() : x.contiguous();

    int seq_len = xc.size(0);
    int batch = xc.size(1);
    int input_size0 = xc.size(2);
    int hidden = h0.size(2);

    auto opts = xc.options();
    auto hn = torch::empty_like(h0);
    torch::Tensor layer_in = xc;

    const int threads = 256;
    const int bh = batch * hidden;
    const int blocks_bh = (bh + threads - 1) / threads;

    for (int l = 0; l < num_layers; ++l) {
        int in_size = (l == 0) ? input_size0 : hidden * 2;

        auto fwd = torch::empty({seq_len, batch, hidden}, opts);
        auto rev = torch::empty({seq_len, batch, hidden}, opts);
        auto layer_out = torch::empty({seq_len, batch, hidden * 2}, opts);

        int base = l * 8;

        gru_direction_kernel<<<blocks_bh, threads>>>(
            layer_in.data_ptr<float>(),
            h0.data_ptr<float>(),
            weights[base + 0].data_ptr<float>(),
            weights[base + 1].data_ptr<float>(),
            weights[base + 2].data_ptr<float>(),
            weights[base + 3].data_ptr<float>(),
            fwd.data_ptr<float>(),
            hn.data_ptr<float>(),
            seq_len, batch, in_size, hidden, 0,
            (l * 2) * batch * hidden
        );

        gru_direction_kernel<<<blocks_bh, threads>>>(
            layer_in.data_ptr<float>(),
            h0.data_ptr<float>(),
            weights[base + 4].data_ptr<float>(),
            weights[base + 5].data_ptr<float>(),
            weights[base + 6].data_ptr<float>(),
            weights[base + 7].data_ptr<float>(),
            rev.data_ptr<float>(),
            hn.data_ptr<float>(),
            seq_len, batch, in_size, hidden, 1,
            (l * 2 + 1) * batch * hidden
        );

        int total = seq_len * batch * hidden;
        int blocks_total = (total + threads - 1) / threads;
        concat_dirs_kernel<<<blocks_total, threads>>>(fwd.data_ptr<float>(), rev.data_ptr<float>(), layer_out.data_ptr<float>(), total, hidden);

        layer_in = layer_out;
    }

    return batch_first ? layer_in.transpose(0, 1).contiguous() : layer_in;
}
"""

gru_ext = load_inline(
    name="custom_gru_bidirectional_fp32",
    cpp_sources=gru_cpp_source,
    cuda_sources=gru_cuda_source,
    functions=["gru_forward_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super().__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout=0,
            bidirectional=True,
        )
        self.num_layers = num_layers
        self.batch_first = batch_first

    def forward(self, x, h0):
        weights = []
        for layer in range(self.num_layers):
            suffix = "" if layer == 0 else str(layer)
            weights.append(getattr(self.gru, "weight_ih_l" + suffix))
            weights.append(getattr(self.gru, "weight_hh_l" + suffix))
            weights.append(getattr(self.gru, "bias_ih_l" + suffix))
            weights.append(getattr(self.gru, "bias_hh_l" + suffix))
            weights.append(getattr(self.gru, "weight_ih_l" + suffix + "_reverse"))
            weights.append(getattr(self.gru, "weight_hh_l" + suffix + "_reverse"))
            weights.append(getattr(self.gru, "bias_ih_l" + suffix + "_reverse"))
            weights.append(getattr(self.gru, "bias_hh_l" + suffix + "_reverse"))
        return gru_ext.gru_forward_cuda(x, h0, weights, self.num_layers, self.batch_first)