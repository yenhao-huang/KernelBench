import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# Custom CUDA kernel for Channel Shuffle
# -----------------------------------------------------------------------------
channel_shuffle_cpp_source = "torch::Tensor channel_shuffle_cuda(torch::Tensor input, int groups);"

channel_shuffle_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void channel_shuffle_kernel(const float* input, float* output, int N, int C, int H, int W, int groups) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    int spatial_size = H * W;
    int n = idx / (C * spatial_size);
    int rem = idx % (C * spatial_size);
    int c_out = rem / spatial_size;
    int spatial_idx = rem % spatial_size;
    int h = spatial_idx / W;
    int w = spatial_idx % W;

    int channels_per_group = C / groups;
    int g = c_out % groups;
    int c_in_group = c_out / groups;
    int c_in = g * channels_per_group + c_in_group;

    int input_idx = ((n * C + c_in) * H + h) * W + w;
    output[idx] = input[input_idx];
}

torch::Tensor channel_shuffle_cuda(torch::Tensor input, int groups) {
    auto N = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);
    auto output = torch::empty_like(input);

    int total = N * C * H * W;
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;

    channel_shuffle_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), N, C, H, W, groups);

    return output;
}
"""

channel_shuffle_op = load_inline(
    name="channel_shuffle_op",
    cpp_sources=channel_shuffle_cpp_source,
    cuda_sources=channel_shuffle_cuda_source,
    functions=["channel_shuffle_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# -----------------------------------------------------------------------------
# Custom CUDA kernel for fused head: adaptive_avg_pool2d + view + fc
# -----------------------------------------------------------------------------
fused_head_cpp_source = "torch::Tensor fused_head_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int H, int W);"

fused_head_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_head_kernel(
    const float* input, const float* weight, const float* bias, float* output,
    int N, int C, int H, int W, int num_classes)
{
    int k = blockIdx.x; // class index
    int n = blockIdx.y; // batch index

    extern __shared__ float shared[];
    float* shared_weight = shared;               // size C
    float* reduction_buf = &shared[C];           // size blockDim.x

    // Load weight row for class k into shared memory
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        shared_weight[i] = weight[k * C + i];
    }
    __syncthreads();

    // Compute partial sum over assigned elements
    float partial_sum = 0.0f;
    int total_spatial = H * W;
    int total_elements = C * total_spatial;
    for (int idx = threadIdx.x; idx < total_elements; idx += blockDim.x) {
        int c = idx / total_spatial;
        int spatial_idx = idx % total_spatial;
        int h = spatial_idx / W;
        int w = spatial_idx % W;
        float val = input[((n * C + c) * H + h) * W + w];
        partial_sum += val * shared_weight[c];
    }

    // Store partial sum to shared memory for reduction
    reduction_buf[threadIdx.x] = partial_sum;
    __syncthreads();

    // Parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduction_buf[threadIdx.x] += reduction_buf[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float sum = reduction_buf[0];
        sum /= (float)(H * W);   // spatial average
        sum += bias[k];
        output[n * num_classes + k] = sum;
    }
}

torch::Tensor fused_head_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int H, int W) {
    auto N = input.size(0);
    auto C = input.size(1);
    auto num_classes = weight.size(0);
    auto output = torch::empty({N, num_classes}, input.options());

    const int block_size = 256;
    dim3 grid(num_classes, N);
    int shared_mem_size = (C + block_size) * sizeof(float);

    fused_head_kernel<<<grid, block_size, shared_mem_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), N, C, H, W, num_classes);

    return output;
}
"""

fused_head_op = load_inline(
    name="fused_head_op",
    cpp_sources=fused_head_cpp_source,
    cuda_sources=fused_head_cuda_source,
    functions=["fused_head_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# -----------------------------------------------------------------------------
# Optimized ShuffleNetUnit (uses custom channel shuffle)
# -----------------------------------------------------------------------------
class ShuffleNetUnitNew(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        super(ShuffleNetUnitNew, self).__init__()
        assert out_channels % 4 == 0
        mid_channels = out_channels // 4

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.groups = groups

        if in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # Custom channel shuffle
        out = channel_shuffle_op.channel_shuffle_cuda(out, self.groups)
        out = F.relu(self.bn3(self.conv3(out)))
        out += self.shortcut(x)
        return out

# -----------------------------------------------------------------------------
# Optimized Model (ShuffleNet)
# -----------------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, num_classes=1000, groups=3, stages_repeats=[3, 7, 3], stages_out_channels=[24, 240, 480, 960]):
        super(ModelNew, self).__init__()

        self.conv1 = nn.Conv2d(3, stages_out_channels[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(stages_out_channels[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage2 = self._make_stage(stages_out_channels[0], stages_out_channels[1], stages_repeats[0], groups)
        self.stage3 = self._make_stage(stages_out_channels[1], stages_out_channels[2], stages_repeats[1], groups)
        self.stage4 = self._make_stage(stages_out_channels[2], stages_out_channels[3], stages_repeats[2], groups)

        self.conv5 = nn.Conv2d(stages_out_channels[3], 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(1024)

        # Keep fc layer for its parameters, but we will not use its forward
        self.fc = nn.Linear(1024, num_classes)

    def _make_stage(self, in_channels, out_channels, repeats, groups):
        layers = []
        layers.append(ShuffleNetUnitNew(in_channels, out_channels, groups))
        for _ in range(1, repeats):
            layers.append(ShuffleNetUnitNew(out_channels, out_channels, groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = F.relu(self.bn5(self.conv5(x)))

        # Fused head: adaptive_avg_pool2d + view + fc
        # x shape: (N, 1024, H, W)
        N, C, H, W = x.shape
        x = x.contiguous()
        x = fused_head_op.fused_head_cuda(x, self.fc.weight, self.fc.bias, H, W)

        return x