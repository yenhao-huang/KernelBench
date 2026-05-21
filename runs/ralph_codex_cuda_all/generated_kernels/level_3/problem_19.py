import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

mobilenet_tail_cpp = """
torch::Tensor mobilenet_tail_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias);
"""

mobilenet_tail_cuda = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

__global__ void avgpool7_kernel(const float* __restrict__ x, float* __restrict__ pooled,
                                int n, int c, int hw) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * c;
    if (idx >= total) return;

    int base = idx * hw;
    float s = 0.0f;

    #pragma unroll
    for (int i = 0; i < 49; ++i) {
        s += x[base + i];
    }

    pooled[idx] = s * 0.02040816326530612f;
}

__global__ void linear_kernel(const float* __restrict__ pooled,
                              const float* __restrict__ weight,
                              const float* __restrict__ bias,
                              float* __restrict__ out,
                              int n, int c, int classes) {
    int cls = blockIdx.x;
    int batch = blockIdx.y;
    int tid = threadIdx.x;

    float acc = 0.0f;
    const float* p = pooled + batch * c;
    const float* w = weight + cls * c;

    for (int k = tid; k < c; k += blockDim.x) {
        acc += p[k] * w[k];
    }

    __shared__ float smem[256];
    smem[tid] = acc;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }

    if (tid == 0) {
        out[batch * classes + cls] = smem[0] + bias[cls];
    }
}

torch::Tensor mobilenet_tail_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    int n = x.size(0);
    int c = x.size(1);
    int h = x.size(2);
    int w = x.size(3);
    int hw = h * w;
    int classes = weight.size(0);

    auto pooled = torch::empty({n, c}, x.options());
    auto out = torch::empty({n, classes}, x.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int threads = 256;
    int total = n * c;
    int blocks = (total + threads - 1) / threads;
    avgpool7_kernel<<<blocks, threads, 0, stream>>>(
        x.data_ptr<float>(), pooled.data_ptr<float>(), n, c, hw
    );

    dim3 grid(classes, n);
    linear_kernel<<<grid, threads, 0, stream>>>(
        pooled.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        out.data_ptr<float>(), n, c, classes
    );

    return out;
}
"""

mobilenet_tail = load_inline(
    name="mobilenet_tail_inline_cuda",
    cpp_sources=mobilenet_tail_cpp,
    cuda_sources=mobilenet_tail_cuda,
    functions=["mobilenet_tail_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000, input_channels=3, alpha=1.0):
        super(ModelNew, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(input_channels, int(32 * alpha), 2),
            conv_dw(int(32 * alpha), int(64 * alpha), 1),
            conv_dw(int(64 * alpha), int(128 * alpha), 2),
            conv_dw(int(128 * alpha), int(128 * alpha), 1),
            conv_dw(int(128 * alpha), int(256 * alpha), 2),
            conv_dw(int(256 * alpha), int(256 * alpha), 1),
            conv_dw(int(256 * alpha), int(512 * alpha), 2),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(1024 * alpha), 2),
            conv_dw(int(1024 * alpha), int(1024 * alpha), 1),
        )
        self.fc = nn.Linear(int(1024 * alpha), num_classes)

    def forward(self, x):
        x = self.model(x)
        return mobilenet_tail.mobilenet_tail_cuda(x, self.fc.weight, self.fc.bias)