import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv1x1_bn_relu_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    float* __restrict__ out,
    int N, int Cin, int H, int W, int Cout, int groups, float eps, bool relu
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * Cout * H * W;
    if (idx >= total) return;

    int s = idx % (H * W);
    int co = (idx / (H * W)) % Cout;
    int n = idx / (Cout * H * W);

    int cin_per_group = Cin / groups;
    int cout_per_group = Cout / groups;
    int g = co / cout_per_group;
    int cin_start = g * cin_per_group;

    float acc = 0.0f;
    const float* xp = x + n * Cin * H * W + cin_start * H * W + s;
    const float* wp = w + co * cin_per_group;

    #pragma unroll 4
    for (int ci = 0; ci < cin_per_group; ++ci) {
        acc += xp[ci * H * W] * wp[ci];
    }

    acc = (acc - mean[co]) * rsqrtf(var[co] + eps) * gamma[co] + beta[co];
    if (relu && acc < 0.0f) acc = 0.0f;
    out[idx] = acc;
}

__global__ void depthwise3x3_bn_shuffle_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    float* __restrict__ out,
    int N, int C, int H, int Wd, int groups, float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * Wd;
    if (idx >= total) return;

    int s = idx % (H * Wd);
    int col = s % Wd;
    int row = s / Wd;
    int c = (idx / (H * Wd)) % C;
    int n = idx / (C * H * Wd);

    float acc = 0.0f;
    const float* base = x + n * C * H * Wd + c * H * Wd;
    const float* kw = w + c * 9;

    for (int ky = -1; ky <= 1; ++ky) {
        int yy = row + ky;
        if (yy < 0 || yy >= H) continue;
        for (int kx = -1; kx <= 1; ++kx) {
            int xx = col + kx;
            if (xx < 0 || xx >= Wd) continue;
            acc += base[yy * Wd + xx] * kw[(ky + 1) * 3 + (kx + 1)];
        }
    }

    acc = (acc - mean[c]) * rsqrtf(var[c] + eps) * gamma[c] + beta[c];

    int channels_per_group = C / groups;
    int shuffled_c = (c % groups) * channels_per_group + (c / groups);
    out[n * C * H * Wd + shuffled_c * H * Wd + s] = acc;
}

__global__ void add_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) out[idx] = a[idx] + b[idx];
}

torch::Tensor conv1x1_bn_relu_cuda(
    torch::Tensor x, torch::Tensor w, torch::Tensor gamma, torch::Tensor beta,
    torch::Tensor mean, torch::Tensor var, int groups, double eps, bool relu
) {
    int N = x.size(0), Cin = x.size(1), H = x.size(2), Wd = x.size(3), Cout = w.size(0);
    auto out = torch::empty({N, Cout, H, Wd}, x.options());
    int total = N * Cout * H * Wd;
    int block = 256;
    int grid = (total + block - 1) / block;
    conv1x1_bn_relu_kernel<<<grid, block>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
        mean.data_ptr<float>(), var.data_ptr<float>(), out.data_ptr<float>(),
        N, Cin, H, Wd, Cout, groups, (float)eps, relu
    );
    return out;
}

torch::Tensor depthwise3x3_bn_shuffle_cuda(
    torch::Tensor x, torch::Tensor w, torch::Tensor gamma, torch::Tensor beta,
    torch::Tensor mean, torch::Tensor var, int groups, double eps
) {
    int N = x.size(0), C = x.size(1), H = x.size(2), Wd = x.size(3);
    auto out = torch::empty_like(x);
    int total = N * C * H * Wd;
    int block = 256;
    int grid = (total + block - 1) / block;
    depthwise3x3_bn_shuffle_kernel<<<grid, block>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
        mean.data_ptr<float>(), var.data_ptr<float>(), out.data_ptr<float>(),
        N, C, H, Wd, groups, (float)eps
    );
    return out;
}

torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b) {
    auto out = torch::empty_like(a);
    int total = a.numel();
    int block = 256;
    int grid = (total + block - 1) / block;
    add_kernel<<<grid, block>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), total);
    return out;
}
"""

cpp_sources = r"""
torch::Tensor conv1x1_bn_relu_cuda(
    torch::Tensor x, torch::Tensor w, torch::Tensor gamma, torch::Tensor beta,
    torch::Tensor mean, torch::Tensor var, int groups, double eps, bool relu
);
torch::Tensor depthwise3x3_bn_shuffle_cuda(
    torch::Tensor x, torch::Tensor w, torch::Tensor gamma, torch::Tensor beta,
    torch::Tensor mean, torch::Tensor var, int groups, double eps
);
torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b);
"""

shufflenet_ops = load_inline(
    name="shufflenet_unit_inline_ops",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=[
        "conv1x1_bn_relu_cuda",
        "depthwise3x3_bn_shuffle_cuda",
        "add_cuda",
    ],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        return x


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        super(ModelNew, self).__init__()
        assert out_channels % 4 == 0
        mid_channels = out_channels // 4
        self.groups = groups

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shuffle = ChannelShuffle(groups)

        if in_channels == out_channels:
            self.shortcut = nn.Sequential()
            self.has_projection = False
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            self.has_projection = True

    def forward(self, x):
        out = shufflenet_ops.conv1x1_bn_relu_cuda(
            x, self.conv1.weight, self.bn1.weight, self.bn1.bias,
            self.bn1.running_mean, self.bn1.running_var, self.groups, self.bn1.eps, True
        )

        out = shufflenet_ops.depthwise3x3_bn_shuffle_cuda(
            out, self.conv2.weight, self.bn2.weight, self.bn2.bias,
            self.bn2.running_mean, self.bn2.running_var, self.groups, self.bn2.eps
        )

        out = shufflenet_ops.conv1x1_bn_relu_cuda(
            out, self.conv3.weight, self.bn3.weight, self.bn3.bias,
            self.bn3.running_mean, self.bn3.running_var, self.groups, self.bn3.eps, True
        )

        if self.has_projection:
            shortcut = shufflenet_ops.conv1x1_bn_relu_cuda(
                x, self.shortcut[0].weight, self.shortcut[1].weight, self.shortcut[1].bias,
                self.shortcut[1].running_mean, self.shortcut[1].running_var, 1, self.shortcut[1].eps, False
            )
        else:
            shortcut = x

        return shufflenet_ops.add_cuda(out, shortcut)