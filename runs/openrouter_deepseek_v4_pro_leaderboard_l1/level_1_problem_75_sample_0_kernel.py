import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

# CUDA source code for custom transposed convolution
conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int N, int C_in, int C_out, int H, int W,
    int H_out, int W_out,
    int kH, int kW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int groups,
    int in_channels_per_group,
    int out_channels_per_group,
    int num_tiles_w)
{
    // Block indices
    int batch = blockIdx.x;
    int group = blockIdx.y;
    int tile_idx = blockIdx.z;
    int tile_h = tile_idx / num_tiles_w;
    int tile_w = tile_idx % num_tiles_w;

    // Thread indices
    int oc_local = threadIdx.x;  // 0 .. out_channels_per_group-1
    int h_local = threadIdx.y;   // 0 .. TILE_H-1
    int w_local = threadIdx.z;   // 0 .. TILE_W-1

    // Shared memory for weight tile of this group
    extern __shared__ float shared_weight[];
    // Layout: [in_channels_per_group][out_channels_per_group][kH][kW]
    // We'll load cooperatively
    int total_weight_elems = in_channels_per_group * out_channels_per_group * kH * kW;
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int total_threads = blockDim.x * blockDim.y * blockDim.z;
    for (int i = tid; i < total_weight_elems; i += total_threads) {
        // Compute weight coordinates
        int tmp = i;
        int kw_idx = tmp % kW; tmp /= kW;
        int kh_idx = tmp % kH; tmp /= kH;
        int oc_idx = tmp % out_channels_per_group; tmp /= out_channels_per_group;
        int ic_idx = tmp;
        // Global weight index
        int global_weight_idx = ((group * in_channels_per_group + ic_idx) * out_channels_per_group + oc_idx) * kH * kW + kh_idx * kW + kw_idx;
        shared_weight[i] = weight[global_weight_idx];
    }
    __syncthreads();

    // Compute output position
    int h_out = tile_h * blockDim.y + h_local;
    int w_out = tile_w * blockDim.z + w_local;
    int oc = group * out_channels_per_group + oc_local;

    if (h_out < H_out && w_out < W_out) {
        float sum = 0.0f;
        int ic_global_start = group * in_channels_per_group;
        for (int ic_local = 0; ic_local < in_channels_per_group; ++ic_local) {
            int ic = ic_global_start + ic_local;
            for (int kh = 0; kh < kH; ++kh) {
                int h_in = h_out + pad_h - kh * dilation_h;
                if (h_in % stride_h != 0) continue;
                h_in /= stride_h;
                if (h_in < 0 || h_in >= H) continue;
                for (int kw = 0; kw < kW; ++kw) {
                    int w_in = w_out + pad_w - kw * dilation_w;
                    if (w_in % stride_w != 0) continue;
                    w_in /= stride_w;
                    if (w_in < 0 || w_in >= W) continue;
                    // Weight index in shared memory
                    int weight_idx = ((ic_local * out_channels_per_group + oc_local) * kH + kh) * kW + kw;
                    float w_val = shared_weight[weight_idx];
                    // Input index
                    int input_idx = ((batch * C_in + ic) * H + h_in) * W + w_in;
                    sum += input[input_idx] * w_val;
                }
            }
        }
        int output_idx = ((batch * C_out + oc) * H_out + h_out) * W_out + w_out;
        output[output_idx] = sum;
    }
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::IntArrayRef stride,
    torch::IntArrayRef padding,
    torch::IntArrayRef dilation,
    int groups)
{
    // Input dimensions
    int N = input.size(0);
    int C_in = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    int kH = weight.size(2);
    int kW = weight.size(3);
    int out_channels_per_group = weight.size(1);
    int C_out = out_channels_per_group * groups;
    int in_channels_per_group = C_in / groups;

    int stride_h = stride[0];
    int stride_w = stride[1];
    int pad_h = padding[0];
    int pad_w = padding[1];
    int dilation_h = dilation[0];
    int dilation_w = dilation[1];

    // Output dimensions
    int H_out = (H - 1) * stride_h - 2 * pad_h + dilation_h * (kH - 1) + 1;
    int W_out = (W - 1) * stride_w - 2 * pad_w + dilation_w * (kW - 1) + 1;

    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    // Tile sizes
    const int TILE_H = 8;
    const int TILE_W = 8;
    int num_tiles_h = (H_out + TILE_H - 1) / TILE_H;
    int num_tiles_w = (W_out + TILE_W - 1) / TILE_W;
    int num_tiles = num_tiles_h * num_tiles_w;

    dim3 block(out_channels_per_group, TILE_H, TILE_W);
    dim3 grid(N, groups, num_tiles);

    size_t shared_mem_size = in_channels_per_group * out_channels_per_group * kH * kW * sizeof(float);

    conv_transpose2d_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, C_out, H, W,
        H_out, W_out,
        kH, kW,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        groups,
        in_channels_per_group,
        out_channels_per_group,
        num_tiles_w
    );

    return output;
}
"""

conv_transpose2d_cpp_source = """
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::IntArrayRef stride,
    torch::IntArrayRef padding,
    torch::IntArrayRef dilation,
    int groups);
"""

# Compile the inline CUDA code
conv_transpose2d_cuda_module = load_inline(
    name="conv_transpose2d_cuda",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: tuple = (1, 1), padding: tuple = (0, 0),
                 dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        # Weight parameter: shape (in_channels, out_channels // groups, kH, kW)
        self.weight = nn.Parameter(
            torch.Tensor(in_channels, out_channels // groups, kernel_size[0], kernel_size[1])
        )
        if bias:
            self.bias_param = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias_param', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = conv_transpose2d_cuda_module.conv_transpose2d_cuda(
            x, self.weight, self.stride, self.padding, self.dilation, self.groups
        )
        if self.bias_param is not None:
            out += self.bias_param.view(1, -1, 1, 1)
        return out