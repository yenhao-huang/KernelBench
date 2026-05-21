import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

final_pool_linear_cpp_source = """
torch::Tensor final_pool_linear_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias);
"""

final_pool_linear_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void final_pool_linear_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int N,
    int C,
    int H,
    int W,
    int K
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * K;
    if (idx >= total) return;

    int n = idx / K;
    int k = idx - n * K;
    int spatial = H * W;
    float inv_spatial = 1.0f / (float)spatial;
    float acc = bias[k];

    for (int c = 0; c < C; ++c) {
        const float* xptr = x + ((n * C + c) * spatial);
        float sum = 0.0f;

        for (int s = 0; s < spatial; ++s) {
            float v = xptr[s];
            sum += v > 0.0f ? v : 0.0f;
        }

        acc += (sum * inv_spatial) * weight[k * C + c];
    }

    out[idx] = acc;
}

torch::Tensor final_pool_linear_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    const int N = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int K = weight.size(0);

    auto out = torch::empty({N, K}, x.options());

    const int threads = 128;
    const int blocks = (N * K + threads - 1) / threads;

    final_pool_linear_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W, K
    );

    return out;
}
"""

final_pool_linear = load_inline(
    name="kb_densenet_final_pool_linear",
    cpp_sources=final_pool_linear_cpp_source,
    cuda_sources=final_pool_linear_cuda_source,
    functions=["final_pool_linear_cuda"],
    verbose=False,
)


class DenseBlockNew(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_features = num_input_features + i * growth_rate
            layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(in_features),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
                    nn.Dropout(0.0),
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(x)
            features.append(new_feature)
            x = torch.cat(features, 1)
        return x


class TransitionLayerNew(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super().__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.transition(x)


class ModelNew(nn.Module):
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        num_features = 64
        block_layers = [6, 12, 24, 16]

        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        for i, num_layers in enumerate(block_layers):
            block = DenseBlockNew(num_layers, num_features, growth_rate)
            self.dense_blocks.append(block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_layers) - 1:
                transition = TransitionLayerNew(num_features, num_features // 2)
                self.transition_layers.append(transition)
                num_features = num_features // 2

        self.final_bn = nn.BatchNorm2d(num_features)
        self.classifier = nn.Linear(num_features, num_classes)
        self.final_pool_linear = final_pool_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)

        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)

        x = self.final_bn(x)
        return self.final_pool_linear.final_pool_linear_cuda(
            x.contiguous(),
            self.classifier.weight.contiguous(),
            self.classifier.bias.contiguous(),
        )