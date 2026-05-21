import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void conv3d_stats_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y,
    float* __restrict__ sums,
    float* __restrict__ sqs,
    int N, int Cin, int D, int H, int Wd,
    int Cout, int K, int OD, int OH, int OW,
    int groups
) {
    long long total = (long long)N * Cout * OD * OH * OW;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int ow = idx % OW;
    int t = idx / OW;
    int oh = t % OH;
    t /= OH;
    int od = t % OD;
    t /= OD;
    int co = t % Cout;
    int n = t / Cout;

    float acc = b[co];

    for (int ci = 0; ci < Cin; ++ci) {
        for (int kz = 0; kz < K; ++kz) {
            int iz = od + kz;
            for (int ky = 0; ky < K; ++ky) {
                int iy = oh + ky;
                for (int kx = 0; kx < K; ++kx) {
                    int ix = ow + kx;
                    long long xidx = (((long long)n * Cin + ci) * D + iz) * H * Wd + iy * Wd + ix;
                    long long widx = (((long long)co * Cin + ci) * K + kz) * K * K + ky * K + kx;
                    acc += x[xidx] * w[widx];
                }
            }
        }
    }

    y[idx] = acc;
    int g = co * groups / Cout;
    int sg = n * groups + g;
    atomicAdd(&sums[sg], acc);
    atomicAdd(&sqs[sg], acc * acc);
}

__global__ void gn_mean_kernel(
    const float* __restrict__ y,
    const float* __restrict__ sums,
    const float* __restrict__ sqs,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    int N, int Cout, int OD, int OH, int OW,
    int groups
) {
    int n = blockIdx.x;
    int tid = threadIdx.x;
    int group_size = Cout / groups;
    int spatial = OD * OH * OW;
    int elems = Cout * spatial;
    float local = 0.0f;

    for (int i = tid; i < elems; i += blockDim.x) {
        int s = i % spatial;
        int co = i / spatial;
        int g = co / group_size;
        int sg = n * groups + g;

        float count = (float)(group_size * spatial);
        float mean = sums[sg] / count;
        float var = sqs[sg] / count - mean * mean;
        float inv = rsqrtf(var + 1.0e-5f);

        long long yidx = ((long long)n * Cout + co) * spatial + s;
        local += (y[yidx] - mean) * inv * gamma[co] + beta[co];
    }

    __shared__ float buf[256];
    buf[tid] = local;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) buf[tid] += buf[tid + stride];
        __syncthreads();
    }

    if (tid == 0) out[n] = buf[0] / (float)elems;
}

torch::Tensor conv3d_gn_mean_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gamma,
    torch::Tensor beta,
    int64_t groups
) {
    int N = x.size(0);
    int Cin = x.size(1);
    int D = x.size(2);
    int H = x.size(3);
    int Wd = x.size(4);
    int Cout = weight.size(0);
    int K = weight.size(2);
    int OD = D - K + 1;
    int OH = H - K + 1;
    int OW = Wd - K + 1;

    auto y = torch::empty({N, Cout, OD, OH, OW}, x.options());
    auto sums = torch::empty({N, (int)groups}, x.options());
    auto sqs = torch::empty({N, (int)groups}, x.options());
    auto out = torch::empty({N}, x.options());

    cudaMemset(sums.data_ptr<float>(), 0, N * groups * sizeof(float));
    cudaMemset(sqs.data_ptr<float>(), 0, N * groups * sizeof(float));

    long long total = (long long)N * Cout * OD * OH * OW;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);

    conv3d_stats_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        y.data_ptr<float>(),
        sums.data_ptr<float>(),
        sqs.data_ptr<float>(),
        N, Cin, D, H, Wd, Cout, K, OD, OH, OW, (int)groups
    );

    gn_mean_kernel<<<N, 256>>>(
        y.data_ptr<float>(),
        sums.data_ptr<float>(),
        sqs.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        out.data_ptr<float>(),
        N, Cout, OD, OH, OW, (int)groups
    );

    return out;
}
"""

cpp_sources = r"""
torch::Tensor conv3d_gn_mean_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gamma,
    torch::Tensor beta,
    int64_t groups
);
"""

_ext = load_inline(
    name="conv3d_gn_mean_inline",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["conv3d_gn_mean_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.num_groups = num_groups
        self._ext = _ext

    def forward(self, x):
        return self._ext.conv3d_gn_mean_cuda(
            x,
            self.conv.weight,
            self.conv.bias,
            self.group_norm.weight,
            self.group_norm.bias,
            self.num_groups,
        )