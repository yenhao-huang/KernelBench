import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

# Custom CUDA code for transposed 1D convolution
conv_transpose1d_cpp_source = """
torch::Tensor conv_transpose1d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                    int64_t stride, int64_t padding, int64_t dilation,
                                    int64_t output_padding, int64_t groups);
"""

conv_transpose1d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose1d_kernel(const float* input, const float* weight, float* output,
                                        int batch_size, int in_channels, int out_channels,
                                        int in_length, int out_length, int out_length_effective,
                                        int kernel_size, int stride, int padding, int dilation,
                                        int output_padding, bool use_bias, const float* bias) {

    int b_oc = blockIdx.x;
    int b = b_oc / out_channels;
    int oc = b_oc % out_channels;
    int tid = threadIdx.x;
    int block_start = blockIdx.y * blockDim.x;
    int p = block_start + tid;

    if (p >= out_length_effective) return;

    extern __shared__ float weight_shared[];
    int total_weight_elems = in_channels * kernel_size;

    // cooperative load of weights for this output channel
    for (int i = tid; i < total_weight_elems; i += blockDim.x) {
        weight_shared[i] = weight[oc * total_weight_elems + i];
    }
    __syncthreads();

    float val = 0.0f;
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int k = 0; k < kernel_size; ++k) {
            int idx = p + padding - k * dilation;
            if (idx % stride != 0) continue;
            int n = idx / stride;
            if (n >= 0 && n < in_length) {
                float input_val = input[b * in_channels * in_length + ic * in_length + n];
                float w = weight_shared[ic * kernel_size + k];
                val += input_val * w;
            }
        }
    }

    if (use_bias) {
        val += bias[oc];
    }

    output[b * out_channels * out_length + oc * out_length + p] = val;
}

torch::Tensor conv_transpose1d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                    int64_t stride, int64_t padding, int64_t dilation,
                                    int64_t output_padding, int64_t groups) {
    TORCH_CHECK(groups == 1, "Only groups=1 is currently supported.");
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto in_length = input.size(2);
    const auto out_channels = weight.size(0);
    const auto kernel_size = weight.size(2);

    int64_t out_length = (in_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    int64_t out_length_effective = out_length - output_padding;

    auto output = torch::zeros({batch_size, out_channels, out_length}, input.options());

    const int threads = 512;
    const int blocks_x = batch_size * out_channels;
    const int blocks_y = (out_length_effective + threads - 1) / threads;
    dim3 grid(blocks_x, blocks_y);
    dim3 block(threads);

    bool use_bias = bias.defined() && bias.numel() > 0;
    const float* bias_ptr = use_bias ? bias.data_ptr<float>() : nullptr;

    size_t shared_mem_size = in_channels * kernel_size * sizeof(float);

    conv_transpose1d_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
        static_cast<int>(batch_size), static_cast<int>(in_channels), static_cast<int>(out_channels),
        static_cast<int>(in_length), static_cast<int>(out_length), static_cast<int>(out_length_effective),
        static_cast<int>(kernel_size), static_cast<int>(stride), static_cast<int>(padding),
        static_cast<int>(dilation), static_cast<int>(output_padding), use_bias, bias_ptr
    );

    return output;
}
"""

# Compile the inline CUDA code
custom_conv_transpose1d = load_inline(
    name="custom_conv_transpose1d",
    cpp_sources=conv_transpose1d_cpp_source,
    cuda_sources=conv_transpose1d_cuda_source,
    functions=["conv_transpose1d_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[],
)


class ModelNew(nn.Module):
    """
    Optimized transposed 1D convolution using a custom CUDA kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, output_padding: int = 0,
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.dilation = 1  # default as in nn.ConvTranspose1d

        # Weight parameter
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialization matching PyTorch's ConvTranspose1d defaults
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        # Store the custom op
        self.custom_conv = custom_conv_transpose1d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        bias = self.bias if self.bias is not None else torch.empty(0, device=x.device)
        return self.custom_conv.conv_transpose1d_cuda(
            x, self.weight, bias,
            self.stride, self.padding, self.dilation,
            self.output_padding, self.groups
        )