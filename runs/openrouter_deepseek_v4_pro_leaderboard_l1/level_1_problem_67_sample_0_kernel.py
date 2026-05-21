import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for 1D convolution (FP32, no padding, stride=1, dilation=1, groups=1, no bias)
conv1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_L 128

__global__ void conv1d_kernel(
    const float* __restrict__ x,      // input: (N, C_in, L)
    const float* __restrict__ weight, // weight: (C_out, C_in, K)
    float* __restrict__ out,          // output: (N, C_out, L_out)
    int N, int C_in, int C_out, int L, int L_out,
    int K, int stride, int padding, int dilation,
    int tile_w)
{
    int n = blockIdx.x;
    int c_out = blockIdx.y;
    int tile_idx = blockIdx.z;

    int l_start = tile_idx * TILE_L;

    int tid = threadIdx.x;

    // dynamic shared memory:
    // - input tile: C_in * tile_w floats
    // - weight tile: C_in * K floats
    extern __shared__ float shm[];
    float* input_tile = shm;
    float* weight_tile = shm + C_in * tile_w;

    int input_tile_total = C_in * tile_w;
    int weight_total = C_in * K;

    // Load input tile cooperatively
    for (int i = tid; i < input_tile_total; i += blockDim.x) {
        int c = i / tile_w;
        int l = i % tile_w;
        int global_l = l_start * stride - padding + l;
        float val = 0.0f;
        if (global_l >= 0 && global_l < L) {
            val = x[n * C_in * L + c * L + global_l];
        }
        input_tile[i] = val;
    }

    // Load weight tile for this output channel
    const float* w_ptr = weight + c_out * C_in * K;
    for (int i = tid; i < weight_total; i += blockDim.x) {
        weight_tile[i] = w_ptr[i];
    }

    __syncthreads();

    // Compute output for this thread
    int l_out = l_start + tid;
    if (l_out < L_out) {
        float sum = 0.0f;
        // Loop over input channels and kernel elements
        for (int c = 0; c < C_in; ++c) {
            // base index in input tile for this c and output position (k=0)
            int base = c * tile_w + tid * stride;
            for (int k = 0; k < K; ++k) {
                sum += input_tile[base + k * dilation] * weight_tile[c * K + k];
            }
        }
        out[n * C_out * L_out + c_out * L_out + l_out] = sum;
    }
}

torch::Tensor conv1d_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation,
    int groups,
    bool bias_on,
    torch::Tensor bias)
{
    int N = x.size(0);
    int C_in = x.size(1);
    int L = x.size(2);
    int C_out = weight.size(0);
    int K = weight.size(2);

    int L_out = (L + 2 * padding - dilation * (K - 1) - 1) / stride + 1;

    auto out = torch::empty({N, C_out, L_out}, x.options());

    int tile_w = (TILE_L - 1) * stride + (K - 1) * dilation + 1;
    int shm_size = (C_in * tile_w + C_in * K) * sizeof(float);

    dim3 grid(N, C_out, (L_out + TILE_L - 1) / TILE_L);
    dim3 block(TILE_L, 1, 1);

    conv1d_kernel<<<grid, block, shm_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C_in, C_out, L, L_out,
        K, stride, padding, dilation,
        tile_w
    );

    // Add bias if required (handles FP32)
    if (bias_on) {
        out.add_(bias.view({1, C_out, 1}));
    }

    return out;
}
"""

conv1d_cpp_source = "torch::Tensor conv1d_cuda(torch::Tensor x, torch::Tensor weight, int stride, int padding, int dilation, int groups, bool bias_on, torch::Tensor bias);"

# Compile the inline CUDA code
conv1d_module = load_inline(
    name="conv1d_cuda",
    cpp_sources=conv1d_cpp_source,
    cuda_sources=conv1d_source,
    functions=["conv1d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    """
    Performs a standard 1D convolution operation, optimized with a custom CUDA kernel.
    (Fixed: stride=1, padding=0, dilation=1, groups=1 for the custom path)
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()

        # The custom kernel currently only supports this configuration.
        assert groups == 1, "Custom kernel only supports groups=1"
        assert dilation == 1, "Custom kernel only supports dilation=1"
        assert padding == 0, "Custom kernel only supports padding=0"
        assert stride == 1, "Custom kernel only supports stride=1"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias_on = bias

        # Weight initialization (matching nn.Conv1d default: kaiming uniform)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size)
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        bias_tensor = self.bias if self.bias is not None else torch.Tensor()
        return conv1d_module.conv1d_cuda(
            x, self.weight,
            self.stride, self.padding, self.dilation, self.groups,
            self.bias_on, bias_tensor
        )