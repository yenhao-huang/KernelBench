import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

linear_cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void linear_kernel(const float* __restrict__ x,
                              const float* __restrict__ w,
                              const float* __restrict__ b,
                              float* __restrict__ out,
                              int batch,
                              int in_features,
                              int out_features) {
    int row = blockIdx.y;
    int col = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float smem[256];
    float acc = 0.0f;

    for (int k = tid; k < in_features; k += blockDim.x) {
        acc += x[row * in_features + k] * w[col * in_features + k];
    }

    smem[tid] = acc;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }

    if (tid == 0) {
        out[row * out_features + col] = smem[0] + b[col];
    }
}

torch::Tensor linear_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
    int batch = x.size(0);
    int in_features = x.size(1);
    int out_features = w.size(0);

    auto out = torch::empty({batch, out_features}, x.options());

    dim3 block(256);
    dim3 grid(out_features, batch);

    linear_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        batch,
        in_features,
        out_features
    );

    return out;
}
"""

linear_cpp_source = "torch::Tensor linear_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b);"

linear_ext = load_inline(
    name="mobilenetv2_custom_linear_ext",
    cpp_sources=linear_cpp_source,
    cuda_sources=linear_cuda_source,
    functions=["linear_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()

        def _make_divisible(v, divisor, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        def _inverted_residual_block(inp, oup, stride, expand_ratio):
            hidden_dim = int(inp * expand_ratio)
            layers = []
            if expand_ratio != 1:
                layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
                layers.append(nn.BatchNorm2d(hidden_dim))
                layers.append(nn.ReLU6(inplace=True))

            layers.extend([
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ])
            return nn.Sequential(*layers)

        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = [
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True),
        ]

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(_inverted_residual_block(input_channel, output_channel, stride, t))
                input_channel = output_channel

        features.append(nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False))
        features.append(nn.BatchNorm2d(last_channel))
        features.append(nn.ReLU6(inplace=True))
        features.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(last_channel, num_classes),
        )
        self.linear_ext = linear_ext

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.linear_ext.linear_cuda(x, self.classifier[1].weight, self.classifier[1].bias)