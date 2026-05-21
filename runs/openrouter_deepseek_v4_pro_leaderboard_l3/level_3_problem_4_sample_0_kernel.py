import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Fused ReLU + MaxPool2d CUDA source
fused_relu_maxpool2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_relu_maxpool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (idx >= total) return;

    int n = idx / (C * H_out * W_out);
    int tmp = idx % (C * H_out * W_out);
    int c = tmp / (H_out * W_out);
    tmp = tmp % (H_out * W_out);
    int oh = tmp / W_out;
    int ow = tmp % W_out;

    float max_val = 0.0f;
    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int h_in = oh * stride_h + kh - pad_h;
            int w_in = ow * stride_w + kw - pad_w;
            if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                float val = input[((n * C + c) * H_in + h_in) * W_in + w_in];
                if (val > max_val) max_val = val;
            }
        }
    }
    output[((n * C + c) * H_out + oh) * W_out + ow] = max_val;
}

torch::Tensor fused_relu_maxpool2d_cuda(
    torch::Tensor input,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding)
{
    TORCH_CHECK(input.dim() == 4, "Input must be 4-dimensional (N, C, H, W)");
    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    int kernel_h = kernel_size;
    int kernel_w = kernel_size;
    int stride_h = stride;
    int stride_w = stride;
    int pad_h = padding;
    int pad_w = padding;

    int H_out = (H + 2 * pad_h - kernel_h) / stride_h + 1;
    int W_out = (W + 2 * pad_w - kernel_w) / stride_w + 1;

    auto output = torch::empty({N, C, H_out, W_out}, input.options());

    int total = N * C * H_out * W_out;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    fused_relu_maxpool2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W, H_out, W_out,
        kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);

    return output;
}
"""

fused_relu_maxpool2d_cpp_source = "torch::Tensor fused_relu_maxpool2d_cuda(torch::Tensor input, int64_t kernel_size, int64_t stride, int64_t padding);"

# Compile the inline CUDA code
fused_relu_maxpool2d_op = load_inline(
    name="fused_relu_maxpool2d",
    cpp_sources=fused_relu_maxpool2d_cpp_source,
    cuda_sources=fused_relu_maxpool2d_source,
    functions=["fused_relu_maxpool2d_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[],
)


class ModelNew(nn.Module):
    def __init__(self, num_classes):
        super(ModelNew, self).__init__()
        # Convolutional layers remain standard
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

        self.fused_pool = fused_relu_maxpool2d_op  # reference to compiled op

    def forward(self, x):
        # First convolution + fused ReLU+MaxPool
        x = self.conv1(x)
        x = self.fused_pool.fused_relu_maxpool2d_cuda(x.contiguous(), 2, 2, 0)

        # Second convolution + fused ReLU+MaxPool
        x = self.conv2(x)
        x = self.fused_pool.fused_relu_maxpool2d_cuda(x.contiguous(), 2, 2, 0)

        # Flatten
        x = x.view(-1, 16 * 5 * 5)

        # Fully connected layers with activation (kept unfused for simplicity, but could be replaced as well)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x