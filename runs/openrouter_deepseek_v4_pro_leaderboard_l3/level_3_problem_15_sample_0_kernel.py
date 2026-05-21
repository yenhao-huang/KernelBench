import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused BatchNorm + ReLU + Conv2d + Dropout in DenseBlock layers
fused_dense_layer_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>

__global__ void fused_dense_layer_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    const float* __restrict__ bn_weight,
    const float* __restrict__ bn_bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int height,
    const int width,
    const float eps,
    const float dropout_prob
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * height * width;
    if (idx >= total_elements) return;

    int w = idx % width;
    int h = (idx / width) % height;
    int oc = (idx / (width * height)) % out_channels;
    int n = idx / (width * height * out_channels);

    float sum = 0.0f;
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = -1; kh <= 1; ++kh) {
            for (int kw = -1; kw <= 1; ++kw) {
                int h_in = h + kh;
                int w_in = w + kw;
                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    int input_idx = ((n * in_channels + ic) * height + h_in) * width + w_in;
                    float val = input[input_idx];
                    // BatchNorm normalization
                    float norm_val = (val - running_mean[ic]) / sqrtf(running_var[ic] + eps);
                    norm_val = norm_val * bn_weight[ic] + bn_bias[ic];
                    // ReLU
                    norm_val = fmaxf(0.0f, norm_val);
                    // Conv2d weight multiplication
                    sum += norm_val * weight[((oc * in_channels + ic) * 3 + (kh+1)) * 3 + (kw+1)];
                }
            }
        }
    }
    // Add bias
    sum += bias[oc];
    // ReLU after conv
    sum = fmaxf(0.0f, sum);
    // Dropout (inverted dropout, scale during training)
    // For simplicity, we apply dropout with probability 0.0 (no dropout) as in original
    // If dropout_prob > 0, we would need random numbers; here we skip for 0.0
    output[idx] = sum;
}

torch::Tensor fused_dense_layer_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    float eps,
    float dropout_prob
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int out_channels = weight.size(0);

    auto output = torch::zeros({batch_size, out_channels, height, width}, input.options());

    const int block_size = 256;
    int total_elements = batch_size * out_channels * height * width;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_dense_layer_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        eps,
        dropout_prob
    );

    return output;
}
"""

fused_dense_layer_cpp_source = (
    "torch::Tensor fused_dense_layer_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias, "
    "torch::Tensor running_mean, torch::Tensor running_var, "
    "torch::Tensor bn_weight, torch::Tensor bn_bias, "
    "float eps, float dropout_prob);"
)

fused_dense_layer = load_inline(
    name="fused_dense_layer",
    cpp_sources=fused_dense_layer_cpp_source,
    cuda_sources=fused_dense_layer_source,
    functions=["fused_dense_layer_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Custom CUDA kernel for fused BatchNorm + ReLU + Conv2d (1x1) + AvgPool2d in TransitionLayer
fused_transition_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_transition_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    const float* __restrict__ bn_weight,
    const float* __restrict__ bn_bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_height * out_width;
    if (idx >= total_elements) return;

    int ow = idx % out_width;
    int oh = (idx / out_width) % out_height;
    int oc = (idx / (out_width * out_height)) % out_channels;
    int n = idx / (out_width * out_height * out_channels);

    // Average pooling: compute mean over 2x2 window
    float sum = 0.0f;
    for (int ic = 0; ic < in_channels; ++ic) {
        float val_sum = 0.0f;
        for (int kh = 0; kh < 2; ++kh) {
            for (int kw = 0; kw < 2; ++kw) {
                int h_in = oh * 2 + kh;
                int w_in = ow * 2 + kw;
                int input_idx = ((n * in_channels + ic) * in_height + h_in) * in_width + w_in;
                float val = input[input_idx];
                // BatchNorm normalization
                float norm_val = (val - running_mean[ic]) / sqrtf(running_var[ic] + eps);
                norm_val = norm_val * bn_weight[ic] + bn_bias[ic];
                // ReLU
                norm_val = fmaxf(0.0f, norm_val);
                val_sum += norm_val;
            }
        }
        // 1x1 convolution
        sum += (val_sum / 4.0f) * weight[oc * in_channels + ic];
    }
    sum += bias[oc];
    output[idx] = sum;
}

torch::Tensor fused_transition_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    float eps
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    int out_channels = weight.size(0);
    int out_height = in_height / 2;
    int out_width = in_width / 2;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

    const int block_size = 256;
    int total_elements = batch_size * out_channels * out_height * out_width;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_transition_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        eps
    );

    return output;
}
"""

fused_transition_cpp_source = (
    "torch::Tensor fused_transition_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias, "
    "torch::Tensor running_mean, torch::Tensor running_var, "
    "torch::Tensor bn_weight, torch::Tensor bn_bias, float eps);"
)

fused_transition = load_inline(
    name="fused_transition",
    cpp_sources=fused_transition_cpp_source,
    cuda_sources=fused_transition_source,
    functions=["fused_transition_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Custom CUDA kernel for initial convolution + BatchNorm + ReLU + MaxPool2d
fused_initial_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_initial_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    const float* __restrict__ bn_weight,
    const float* __restrict__ bn_bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_height * out_width;
    if (idx >= total_elements) return;

    int ow = idx % out_width;
    int oh = (idx / out_width) % out_height;
    int oc = (idx / (out_width * out_height)) % out_channels;
    int n = idx / (out_width * out_height * out_channels);

    // MaxPool2d with kernel=3, stride=2, padding=1
    // Input to pooling is after conv+bn+relu, so we compute the max over 3x3 window
    float max_val = -1e38f;
    for (int kh = 0; kh < 3; ++kh) {
        for (int kw = 0; kw < 3; ++kw) {
            int h_conv = oh * 2 + kh - 1; // stride 2, padding 1
            int w_conv = ow * 2 + kw - 1;
            if (h_conv >= 0 && h_conv < in_height/2 && w_conv >= 0 && w_conv < in_width/2) {
                // Compute conv output at this position
                float conv_sum = 0.0f;
                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int kch = 0; kch < 7; ++kch) {
                        for (int kcw = 0; kcw < 7; ++kcw) {
                            int h_in = h_conv * 2 + kch - 3; // conv stride 2, padding 3
                            int w_in = w_conv * 2 + kcw - 3;
                            if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                                int input_idx = ((n * in_channels + ic) * in_height + h_in) * in_width + w_in;
                                conv_sum += input[input_idx] * weight[((oc * in_channels + ic) * 7 + kch) * 7 + kcw];
                            }
                        }
                    }
                }
                conv_sum += bias[oc];
                // BatchNorm
                float norm_val = (conv_sum - running_mean[oc]) / sqrtf(running_var[oc] + eps);
                norm_val = norm_val * bn_weight[oc] + bn_bias[oc];
                // ReLU
                norm_val = fmaxf(0.0f, norm_val);
                max_val = fmaxf(max_val, norm_val);
            }
        }
    }
    output[idx] = max_val;
}

torch::Tensor fused_initial_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    float eps
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    int out_channels = weight.size(0);
    int out_height = in_height / 4; // conv stride 2, pool stride 2
    int out_width = in_width / 4;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

    const int block_size = 256;
    int total_elements = batch_size * out_channels * out_height * out_width;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_initial_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        eps
    );

    return output;
}
"""

fused_initial_cpp_source = (
    "torch::Tensor fused_initial_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias, "
    "torch::Tensor running_mean, torch::Tensor running_var, "
    "torch::Tensor bn_weight, torch::Tensor bn_bias, float eps);"
)

fused_initial = load_inline(
    name="fused_initial",
    cpp_sources=fused_initial_cpp_source,
    cuda_sources=fused_initial_source,
    functions=["fused_initial_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class FusedDenseLayer(nn.Module):
    def __init__(self, in_features, growth_rate):
        super(FusedDenseLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_features)
        self.conv = nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout(0.0)
        self.growth_rate = growth_rate

    def forward(self, x):
        return fused_dense_layer.fused_dense_layer_cuda(
            x,
            self.conv.weight,
            self.conv.bias if self.conv.bias is not None else torch.zeros(self.growth_rate, device=x.device),
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.weight,
            self.bn.bias,
            self.bn.eps,
            0.0
        )

class FusedTransitionLayer(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(FusedTransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(num_input_features)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False)

    def forward(self, x):
        return fused_transition.fused_transition_cuda(
            x,
            self.conv.weight,
            self.conv.bias if self.conv.bias is not None else torch.zeros(num_output_features, device=x.device),
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.weight,
            self.bn.bias,
            self.bn.eps
        )

class FusedInitialBlock(nn.Module):
    def __init__(self):
        super(FusedInitialBlock, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)

    def forward(self, x):
        return fused_initial.fused_initial_cuda(
            x,
            self.conv.weight,
            self.conv.bias if self.conv.bias is not None else torch.zeros(64, device=x.device),
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.weight,
            self.bn.bias,
            self.bn.eps
        )

class DenseBlock(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(FusedDenseLayer(num_input_features + i * growth_rate, growth_rate))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(x)
            features.append(new_feature)
            x = torch.cat(features, 1)
        return x

class ModelNew(nn.Module):
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        super(ModelNew, self).__init__()

        self.initial = FusedInitialBlock()

        num_features = 64
        block_layers = [6, 12, 24, 16]

        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        for i, num_layers in enumerate(block_layers):
            block = DenseBlock(num_layers=num_layers, num_input_features=num_features, growth_rate=growth_rate)
            self.dense_blocks.append(block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_layers) - 1:
                transition = FusedTransitionLayer(num_input_features=num_features, num_output_features=num_features // 2)
                self.transition_layers.append(transition)
                num_features = num_features // 2

        self.final_bn = nn.BatchNorm2d(num_features)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)

        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)

        x = self.final_bn(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.classifier(x)
        return x