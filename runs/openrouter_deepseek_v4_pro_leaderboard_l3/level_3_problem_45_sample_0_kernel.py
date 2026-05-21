import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels for fused BatchNorm + Softmax
fused_bn_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void compute_channel_stats_kernel(
    const float* __restrict__ input,
    float* __restrict__ sum,
    float* __restrict__ sum_sq,
    int N, int C, int H, int W,
    int total_per_channel,
    int num_blocks_per_channel)
{
    int c = blockIdx.x;
    int block_id = blockIdx.y;
    int thread_id = threadIdx.x;
    int block_size = blockDim.x;
    
    // Compute chunk size for this block
    int chunk_size = (total_per_channel + num_blocks_per_channel - 1) / num_blocks_per_channel;
    int start = block_id * chunk_size;
    int end = min(start + chunk_size, total_per_channel);
    
    // Shared memory for partial sums: first block_size floats for sum, next for sum_sq
    extern __shared__ float shared[];
    float* s_sum = shared;
    float* s_sum_sq = shared + block_size;
    
    // Initialize shared memory
    s_sum[thread_id] = 0.0f;
    s_sum_sq[thread_id] = 0.0f;
    __syncthreads();
    
    // Each thread processes multiple elements within the chunk
    for (int idx = start + thread_id; idx < end; idx += block_size) {
        // Convert linear index within channel to (n, h, w)
        int n = idx / (H * W);
        int hw = idx % (H * W);
        int h = hw / W;
        int w = hw % W;
        float val = input[n * C * H * W + c * H * W + h * W + w];
        s_sum[thread_id] += val;
        s_sum_sq[thread_id] += val * val;
    }
    __syncthreads();
    
    // Reduce within block
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (thread_id < stride) {
            s_sum[thread_id] += s_sum[thread_id + stride];
            s_sum_sq[thread_id] += s_sum_sq[thread_id + stride];
        }
        __syncthreads();
    }
    
    // Atomic add to global
    if (thread_id == 0) {
        atomicAdd(&sum[c], s_sum[0]);
        atomicAdd(&sum_sq[c], s_sum_sq[0]);
    }
}

__global__ void fused_bn_softmax_kernel(
    const float* __restrict__ input,
    const float* __restrict__ sum,
    const float* __restrict__ sum_sq,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W,
    float eps,
    int total_per_channel)
{
    // Block index: n*C*H + c*H + h
    int block_idx = blockIdx.x;
    int n = block_idx / (C * H);
    int rem = block_idx % (C * H);
    int c = rem / H;
    int h = rem % H;
    
    int thread_id = threadIdx.x;
    int block_size = blockDim.x;
    
    // Compute mean and variance for this channel
    float mean = sum[c] / total_per_channel;
    float var = sum_sq[c] / total_per_channel - mean * mean;
    float inv_std = rsqrtf(var + eps);
    
    // Dynamic shared memory layout:
    // [0, W) : normalized values (vec)
    // [W]     : max value
    // [W+1]   : sum of exp
    extern __shared__ float shared[];
    float* vec = shared;
    float* max_val = &shared[W];
    float* sum_exp = &shared[W + 1];
    
    // Load and normalize elements, store in shared
    for (int w = thread_id; w < W; w += block_size) {
        float val = input[n * C * H * W + c * H * W + h * W + w];
        float norm = (val - mean) * inv_std;
        norm = norm * weight[c] + bias[c];
        vec[w] = norm;
    }
    __syncthreads();
    
    // Find max (single thread for simplicity, W is small)
    if (thread_id == 0) {
        float max_v = vec[0];
        for (int w = 1; w < W; ++w) {
            if (vec[w] > max_v) max_v = vec[w];
        }
        *max_val = max_v;
    }
    __syncthreads();
    
    // Compute sum of exp
    if (thread_id == 0) {
        float sum_e = 0.0f;
        for (int w = 0; w < W; ++w) {
            sum_e += expf(vec[w] - *max_val);
        }
        *sum_exp = sum_e;
    }
    __syncthreads();
    
    // Write softmax output
    for (int w = thread_id; w < W; w += block_size) {
        float softmax_val = expf(vec[w] - *max_val) / (*sum_exp);
        output[n * C * H * W + c * H * W + h * W + w] = softmax_val;
    }
}

torch::Tensor fused_bn_softmax_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps)
{
    const auto N = input.size(0);
    const auto C = input.size(1);
    const auto H = input.size(2);
    const auto W = input.size(3);
    
    auto output = torch::empty_like(input);
    
    int total_per_channel = N * H * W;
    
    // Allocate sum and sum_sq tensors (initialized to zero)
    auto sum = torch::zeros({C}, input.options());
    auto sum_sq = torch::zeros({C}, input.options());
    
    // Launch compute_channel_stats_kernel
    const int block_size = 256;
    int num_blocks_per_channel = (total_per_channel + block_size - 1) / block_size;
    dim3 grid_stats(C, num_blocks_per_channel);
    size_t shared_mem_stats = 2 * block_size * sizeof(float);
    
    compute_channel_stats_kernel<<<grid_stats, block_size, shared_mem_stats>>>(
        input.data_ptr<float>(),
        sum.data_ptr<float>(),
        sum_sq.data_ptr<float>(),
        N, C, H, W,
        total_per_channel,
        num_blocks_per_channel);
    
    // Launch fused_bn_softmax_kernel
    int total_blocks = N * C * H;
    int softmax_block_size = 256;  // each thread handles multiple elements if W > 256
    size_t shared_mem_softmax = (W + 2) * sizeof(float);
    
    fused_bn_softmax_kernel<<<total_blocks, softmax_block_size, shared_mem_softmax>>>(
        input.data_ptr<float>(),
        sum.data_ptr<float>(),
        sum_sq.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        eps,
        total_per_channel);
    
    return output;
}
"""

fused_bn_softmax_cpp_source = "torch::Tensor fused_bn_softmax_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float eps);"

# Compile the inline CUDA code
fused_bn_softmax = load_inline(
    name="fused_bn_softmax",
    cpp_sources=fused_bn_softmax_cpp_source,
    cuda_sources=fused_bn_softmax_source,
    functions=["fused_bn_softmax_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class FusedBNSoftmax(nn.Module):
    """Custom module that fuses BatchNorm2d and Softmax(dim=-1) into a single CUDA kernel."""
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x):
        return fused_bn_softmax.fused_bn_softmax_cuda(x, self.weight, self.bias, self.eps)


class DoubleConv(nn.Module):
    """Optimized DoubleConv using custom fused BatchNorm+Softmax."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.fused1 = FusedBNSoftmax(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.fused2 = FusedBNSoftmax(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.fused1(x)
        x = self.conv2(x)
        x = self.fused2(x)
        return x


class ModelNew(nn.Module):
    """U-Net with custom CUDA operators for BatchNorm+Softmax fusion."""
    def __init__(self, in_channels, out_channels, features):
        super(ModelNew, self).__init__()
        self.encoder1 = DoubleConv(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = DoubleConv(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = DoubleConv(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = DoubleConv(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConv(features * 8, features * 16)

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(features * 16, features * 8)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(features * 8, features * 4)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(features * 4, features * 2)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(features * 2, features)

        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.final_conv(dec1)