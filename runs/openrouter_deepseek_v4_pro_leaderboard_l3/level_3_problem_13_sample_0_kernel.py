import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA source for fused BatchNorm2d + ReLU + Conv2d(1x1) + AvgPool2d(2x2 stride 2)
fused_bn_relu_conv_pool_source_cpp = """
torch::Tensor fused_bn_relu_conv_pool_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor conv_weight,
    float eps);
"""

fused_bn_relu_conv_pool_source_cuda = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_bn_relu_conv_pool_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    const float* __restrict__ conv_weight,
    float* __restrict__ output,
    const int batch,
    const int in_c,
    const int out_c,
    const int H,
    const int W,
    const float eps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_h = H / 2;
    int out_w = W / 2;
    int total = batch * out_c * out_h * out_w;
    if (idx >= total) return;

    // Decode flat index into (b, c_out, oh, ow)
    int tmp = idx;
    int ow = tmp % out_w; tmp /= out_w;
    int oh = tmp % out_h; tmp /= out_h;
    int c_out = tmp % out_c; tmp /= out_c;
    int b = tmp;

    float sum = 0.0f;
    for (int c_in = 0; c_in < in_c; ++c_in) {
        float mean = running_mean[c_in];
        float var = running_var[c_in];
        float inv_std = 1.0f / sqrtf(var + eps);
        float g = gamma[c_in];
        float be = beta[c_in];
        float w = conv_weight[c_out * in_c + c_in]; // row-major weight[C_out][C_in]

        // 2x2 pooling window
        for (int i = 0; i < 2; ++i) {
            int h_idx = oh * 2 + i;
            for (int j = 0; j < 2; ++j) {
                int w_idx = ow * 2 + j;
                float val = input[((b * in_c + c_in) * H + h_idx) * W + w_idx];
                float bn = (val - mean) * inv_std * g + be;
                if (bn > 0.0f) {
                    sum += bn * w;
                }
            }
        }
    }
    output[idx] = sum * 0.25f; // average over 4 spatial locations
}

torch::Tensor fused_bn_relu_conv_pool_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor conv_weight,
    float eps)
{
    AT_ASSERTM(input.is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(input.dtype() == torch::kFloat32, "input must be float32");

    const int batch = input.size(0);
    const int in_c = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    const int out_c = conv_weight.size(0);
    const int out_h = H / 2;
    const int out_w = W / 2;

    auto output = torch::empty({batch, out_c, out_h, out_w}, input.options());

    const int total = batch * out_c * out_h * out_w;
    const int block = 256;
    const int grid = (total + block - 1) / block;

    fused_bn_relu_conv_pool_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        conv_weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, in_c, out_c, H, W, eps);

    return output;
}
"""

# Compile the custom CUDA operator
fused_op = load_inline(
    name="fused_bn_relu_conv_pool",
    cpp_sources=fused_bn_relu_conv_pool_source_cpp,
    cuda_sources=fused_bn_relu_conv_pool_source_cuda,
    functions=["fused_bn_relu_conv_pool_cuda"],
    verbose=False,
    extra_cflags=[],
    extra_ldflags=[],
)


class ModelNew(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        """
        :param num_input_features: The number of input feature maps
        :param num_output_features: The number of output feature maps
        """
        super(ModelNew, self).__init__()
        self.num_input = num_input_features
        self.num_output = num_output_features

        # BatchNorm parameters
        self.gamma = nn.Parameter(torch.ones(num_input_features))
        self.beta = nn.Parameter(torch.zeros(num_input_features))
        self.register_buffer("running_mean", torch.zeros(num_input_features))
        self.register_buffer("running_var", torch.ones(num_input_features))
        self.eps = 1e-5   # default BatchNorm epsilon

        # Conv2d 1x1 weight (no bias, as in original)
        self.conv_weight = nn.Parameter(torch.empty(num_output_features, num_input_features))
        nn.init.kaiming_uniform_(self.conv_weight)  # match default Conv2d init

        self.fused_op = fused_op

    def forward(self, x):
        """
        :param x: Input tensor of shape (batch_size, num_input_features, height, width)
        :return: Downsampled tensor with reduced number of feature maps
        """
        # Ensure input is contiguous for the CUDA kernel
        if not x.is_contiguous():
            x = x.contiguous()
        return self.fused_op.fused_bn_relu_conv_pool_cuda(
            x,
            self.gamma,
            self.beta,
            self.running_mean,
            self.running_var,
            self.conv_weight,
            self.eps,
        )