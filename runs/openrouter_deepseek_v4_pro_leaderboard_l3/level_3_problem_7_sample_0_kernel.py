import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA source for ReLU kernel
relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        output[idx] = val > 0.0f ? val : 0.0f;
    }
}

torch::Tensor relu_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    relu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    return output;
}
"""

relu_cpp_source = "torch::Tensor relu_cuda(torch::Tensor input);"

# CUDA source for fused avgpool + fc kernel
avgpool_fc_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avgpool_fc_kernel(const float* input, const float* weight, const float* bias, float* output,
                                  int N, int C, int H, int W, int num_classes) {
    // Block index determines (sample, class)
    int n = blockIdx.x / num_classes;
    int k = blockIdx.x % num_classes;
    int c = threadIdx.x;

    // Shared memory for reduction, size = C
    extern __shared__ float sdata[];

    // Each thread handles one channel c: compute spatial sum
    float sum = 0.0f;
    if (c < C) {
        // Base pointer for sample n, channel c
        const float* base = input + n * C * H * W + c * H * W;
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                sum += base[h * W + w];
            }
        }
        // Multiply by weight[k, c]
        sdata[c] = sum * weight[k * C + c];
    } else {
        sdata[c] = 0.0f;
    }
    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (c < stride) {
            sdata[c] += sdata[c + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes final result with bias
    if (c == 0) {
        output[n * num_classes + k] = sdata[0] + bias[k];
    }
}

torch::Tensor avgpool_fc_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int H, int W) {
    // Ensure contiguous
    input = input.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();

    auto N = input.size(0);
    auto C = input.size(1);
    auto num_classes = weight.size(0);

    auto output = torch::empty({N, num_classes}, input.options());

    const int threads = C;            // C must be <= 1024
    const int blocks = N * num_classes;

    avgpool_fc_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), N, C, H, W, num_classes);

    return output;
}
"""

avgpool_fc_cpp_source = "torch::Tensor avgpool_fc_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int H, int W);"

# Compile custom extensions
relu_module = load_inline(
    name="relu_cuda",
    cpp_sources=relu_cpp_source,
    cuda_sources=relu_source,
    functions=["relu_cuda"],
    verbose=False
)

avgpool_fc_module = load_inline(
    name="avgpool_fc_cuda",
    cpp_sources=avgpool_fc_cpp_source,
    cuda_sources=avgpool_fc_source,
    functions=["avgpool_fc_cuda"],
    verbose=False
)


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        super(InceptionModule, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3x3, kernel_size=1),
            nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)
        )
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5x5, kernel_size=1),
            nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()

        self.relu = relu_module
        self.avgpool_fc = avgpool_fc_module

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        # Keep the FC layer for its parameters, but use custom fused avgpool+fc instead of forward
        self.fc = nn.Linear(1024, num_classes)

        # Extract FC parameters for the custom kernel
        self.fc_weight = self.fc.weight.data
        self.fc_bias = self.fc.bias.data

    def forward(self, x):
        x = self.maxpool1(self.relu.relu_cuda(self.conv1(x)))
        x = self.relu.relu_cuda(self.conv2(x))
        x = self.maxpool2(self.relu.relu_cuda(self.conv3(x)))

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        # Fused adaptive average pooling + flatten + linear
        H, W = x.shape[2], x.shape[3]
        x = self.avgpool_fc.avgpool_fc_cuda(x, self.fc_weight, self.fc_bias, H, W)

        return x