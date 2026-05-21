import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void softmax_lastdim_kernel(const float* __restrict__ x, float* __restrict__ y, int rows, int W) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    const float* xr = x + row * W;
    float* yr = y + row * W;

    float m = -FLT_MAX;
    for (int i = tid; i < W; i += blockDim.x) {
        float v = xr[i];
        m = fmaxf(m, v);
    }
    smem[tid] = m;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    m = smem[0];

    float sum = 0.0f;
    for (int i = tid; i < W; i += blockDim.x) {
        float e = expf(xr[i] - m);
        yr[i] = e;
        sum += e;
    }
    smem[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float inv = 1.0f / smem[0];

    for (int i = tid; i < W; i += blockDim.x) {
        yr[i] *= inv;
    }
}

__global__ void concat_channel_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int total,
    int C1,
    int C2,
    int H,
    int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int HW = H * W;
    int C = C1 + C2;
    int inner = idx % HW;
    int c = (idx / HW) % C;
    int n = idx / (HW * C);

    if (c < C1) {
        out[idx] = a[(n * C1 + c) * HW + inner];
    } else {
        out[idx] = b[(n * C2 + (c - C1)) * HW + inner];
    }
}

torch::Tensor softmax_lastdim_cuda(torch::Tensor x) {
    auto y = torch::empty_like(x);
    int W = (int)x.size(3);
    int rows = (int)(x.numel() / W);
    int threads = 256;
    softmax_lastdim_kernel<<<rows, threads, threads * sizeof(float)>>>(x.data_ptr<float>(), y.data_ptr<float>(), rows, W);
    return y;
}

torch::Tensor concat_channel_cuda(torch::Tensor a, torch::Tensor b) {
    int N = (int)a.size(0);
    int C1 = (int)a.size(1);
    int C2 = (int)b.size(1);
    int H = (int)a.size(2);
    int W = (int)a.size(3);
    auto out = torch::empty({N, C1 + C2, H, W}, a.options());
    int total = (int)out.numel();
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    concat_channel_kernel<<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), total, C1, C2, H, W);
    return out;
}
"""

cpp_sources = r"""
torch::Tensor softmax_lastdim_cuda(torch::Tensor x);
torch::Tensor concat_channel_cuda(torch::Tensor a, torch::Tensor b);
"""

_unet_ops = load_inline(
    name="unet_softmax_concat_ops",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["softmax_lastdim_cuda", "concat_channel_cuda"],
    verbose=False,
)


class CudaSoftmaxLastDim(nn.Module):
    def forward(self, x):
        return _unet_ops.softmax_lastdim_cuda(x.contiguous())


class DoubleConvNew(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            CudaSoftmaxLastDim(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            CudaSoftmaxLastDim(),
        )

    def forward(self, x):
        return self.double_conv(x)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, features):
        super(ModelNew, self).__init__()
        self.encoder1 = DoubleConvNew(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = DoubleConvNew(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = DoubleConvNew(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = DoubleConvNew(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConvNew(features * 8, features * 16)

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = DoubleConvNew(features * 16, features * 8)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = DoubleConvNew(features * 8, features * 4)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConvNew(features * 4, features * 2)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = DoubleConvNew(features * 2, features)

        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = _unet_ops.concat_channel_cuda(dec4.contiguous(), enc4.contiguous())
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = _unet_ops.concat_channel_cuda(dec3.contiguous(), enc3.contiguous())
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = _unet_ops.concat_channel_cuda(dec2.contiguous(), enc2.contiguous())
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = _unet_ops.concat_channel_cuda(dec1.contiguous(), enc1.contiguous())
        dec1 = self.decoder1(dec1)

        return self.final_conv(dec1)