import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Fused BatchNorm (inference mode) + ReLU kernel
# In inference, BatchNorm is just: y = gamma * (x - mean) / sqrt(var + eps) + beta
# We can fuse this with ReLU.
fused_bn_relu_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_bn_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int num_channels,
    int spatial_size,
    float eps) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_channels * spatial_size;

    if (idx < total_elements) {
        // Calculate channel index
        // idx = b * (C * S) + c * S + s
        int spatial_idx = idx % spatial_size;
        int channel_idx = (idx / spatial_size) % num_channels;
        
        float x = input[idx];
        float mean = running_mean[channel_idx];
        float var = running_var[channel_idx];
        float gamma = weight[channel_idx];
        float beta = bias[channel_idx];
        
        // BatchNorm formula
        float inv_std = 1.0f / sqrtf(var + eps);
        float val = gamma * (x - mean) * inv_std + beta;
        
        // ReLU
        output[idx] = val > 0.0f ? val : 0.0f;
    }
}

torch::Tensor fused_bn_relu_cuda(
    torch::Tensor input,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps) 
{
    auto batch_size = input.size(0);
    auto num_channels = input.size(1);
    auto spatial_size = input.size(2) * input.size(3);
    auto output = torch::empty_like(input);

    int total_elements = batch_size * num_channels * spatial_size;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_bn_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_channels,
        spatial_size,
        eps
    );

    return output;
}
"""

fused_bn_relu_cpp_source = "torch::Tensor fused_bn_relu_cuda(torch::Tensor input, torch::Tensor running_mean, torch::Tensor running_var, torch::Tensor weight, torch::Tensor bias, float eps);"

fused_bn_relu = load_inline(
    name="fused_bn_relu",
    cpp_sources=fused_bn_relu_cpp_source,
    cuda_sources=fused_bn_relu_cuda_source,
    functions=["fused_bn_relu_cuda"],
    verbose=False
)

class FusedBNReLU(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        # Use the custom fused kernel during inference
        if not self.training:
            return fused_bn_relu.fused_bn_relu_cuda(
                x, self.bn.running_mean, self.bn.running_var, 
                self.bn.weight, self.bn.bias, self.eps
            )
        else:
            # Fallback to standard for training
            return F.relu(self.bn(x), inplace=True)

    def train(self, mode=True):
        super().train(mode)
        return self

# Now we integrate this into the model.
# We'll also use the DenseBlock optimization (pre-allocation).

class DenseBlockNew(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super(DenseBlockNew, self).__init__()
        self.num_layers = num_layers
        self.num_input_features = num_input_features
        self.growth_rate = growth_rate
        self.total_output_features = num_input_features + num_layers * growth_rate
        
        layers = []
        for i in range(num_layers):
            layers.append(self._make_layer(num_input_features + i * growth_rate, growth_rate))
        self.layers = nn.ModuleList(layers)

    def _make_layer(self, in_features: int, growth_rate: int):
        return nn.Sequential(
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout(0.0)
        )

    def forward(self, x):
        B, C_in, H, W = x.shape
        out = torch.empty((B, self.total_output_features, H, W), dtype=x.dtype, device=x.device)
        out[:, :C_in, :, :] = x
        
        current_channels = C_in
        for layer in self.layers:
            layer_input = out[:, :current_channels, :, :]
            new_feat = layer(layer_input)
            out[:, current_channels:current_channels + self.growth_rate, :, :] = new_feat
            current_channels += self.growth_rate
        return out

class TransitionLayerNew(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super(TransitionLayerNew, self).__init__()
        # Using FusedBNReLU for the transition layer
        self.bn = FusedBNReLU(num_input_features)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # The original order was BN -> ReLU -> Conv -> Pool
        # Our FusedBNReLU does BN + ReLU.
        x = self.bn(x)
        x = self.conv(x)
        x = self.pool(x)
        return x

class ModelNew(nn.Module):
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        super(ModelNew, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        num_features = 64
        block_layers = [6, 12, 48, 32]

        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        for i, num_layers in enumerate(block_layers):
            block = DenseBlockNew(num_layers=num_layers, num_input_features=num_features, growth_rate=growth_rate)
            self.dense_blocks.append(block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_layers) - 1:
                transition = TransitionLayerNew(num_input_features=num_features, num_output_features=num_features // 2)
                self.transition_layers.append(transition)
                num_features = num_features // 2

        self.final_bn = FusedBNReLU(num_features)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)

        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)

        x = self.final_bn(x)
        # Note: FusedBNReLU already applied ReLU.
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.classifier(x)
        return x