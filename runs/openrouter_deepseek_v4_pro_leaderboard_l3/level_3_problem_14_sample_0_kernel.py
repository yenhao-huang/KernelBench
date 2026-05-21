import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------
# Custom CUDA kernel: fused BN -> ReLU -> 3x3 convolution
# ------------------------------------------------------------
fused_bn_relu_conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_H 16
#define TILE_W 16
#define INNER_C 8
#define PAD 32

__global__ void fused_bn_relu_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bn_weight,
    const float* __restrict__ bn_bias,
    const float* __restrict__ bn_running_mean,
    const float* __restrict__ bn_running_var,
    const float* __restrict__ conv_weight,
    float* __restrict__ output,
    int N, int C_in, int H, int W, int C_out,
    float eps)
{
    // Shared memory for a tile of input channels (already BN+ReLU applied)
    __shared__ float shared_input[INNER_C][TILE_H+2][PAD];
    // Shared memory for the corresponding weight slice
    __shared__ float shared_weight[INNER_C][3][3];

    // Block indexing
    int bc = blockIdx.x;                // combined batch + output channel index
    int tile_idx = blockIdx.y;          // flattened output spatial tile index

    int n = bc / C_out;
    int out_c = bc % C_out;

    int num_tiles_w = (W + TILE_W - 1) / TILE_W;
    int tile_h = tile_idx / num_tiles_w;
    int tile_w = tile_idx % num_tiles_w;

    int out_h_start = tile_h * TILE_H;
    int out_w_start = tile_w * TILE_W;

    // Thread‑local output position inside the tile
    int th = threadIdx.y;
    int tw = threadIdx.x;
    int out_h = out_h_start + th;
    int out_w = out_w_start + tw;
    bool valid = (out_h < H && out_w < W);

    // Accumulator for this output pixel
    float accum = 0.0f;

    // Tile offset helpers
    int load_h = out_h_start - 1;  // first y of halo
    int load_w = out_w_start - 1;  // first x of halo
    int tile_load_h = TILE_H + 2;
    int tile_load_w = TILE_W + 2;
    int total_load = tile_load_h * tile_load_w;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * blockDim.y;

    // Loop over input channel segments
    for (int c_start = 0; c_start < C_in; c_start += INNER_C) {
        // 1) Load input patch for this channel segment, apply BN+ReLU -> shared_input
        for (int idx = tid; idx < total_load * INNER_C; idx += num_threads) {
            int c_local = idx / total_load;
            int c_global = c_start + c_local;
            int pos = idx % total_load;
            int ly = pos / tile_load_w;
            int lx = pos % tile_load_w;

            float val = 0.0f;
            if (c_global < C_in) {
                int src_h = load_h + ly;
                int src_w = load_w + lx;
                if (src_h >= 0 && src_h < H && src_w >= 0 && src_w < W) {
                    // Read input directly
                    int in_idx = ((n * C_in + c_global) * H + src_h) * W + src_w;
                    val = input[in_idx];
                } else {
                    val = 0.0f;
                }
                // BatchNorm
                float mean = bn_running_mean[c_global];
                float var = bn_running_var[c_global];
                float inv_std = rsqrtf(var + eps);
                val = (val - mean) * inv_std * bn_weight[c_global] + bn_bias[c_global];
                // ReLU
                val = fmaxf(val, 0.0f);
            }
            // Pad x dimension to avoid bank conflicts
            shared_input[c_local][ly][lx] = val;
        }

        // 2) Load weight slice for this output channel and channel segment
        for (int idx = tid; idx < INNER_C * 9; idx += num_threads) {
            int c_local = idx / 9;
            int rem = idx % 9;
            int ky = rem / 3;
            int kx = rem % 3;
            int c_global = c_start + c_local;
            float w = 0.0f;
            if (c_global < C_in) {
                int w_idx = ((out_c * C_in + c_global) * 3 + ky) * 3 + kx;
                w = conv_weight[w_idx];
            }
            shared_weight[c_local][ky][kx] = w;
        }

        __syncthreads();

        // 3) Compute partial convolution
        if (valid) {
            for (int c_local = 0; c_local < INNER_C; ++c_local) {
                int c_global = c_start + c_local;
                if (c_global >= C_in) break;
                for (int ky = 0; ky < 3; ++ky) {
                    for (int kx = 0; kx < 3; ++kx) {
                        // input_y = out_h + ky - 1  inside the halo (offset = th + ky)
                        // input_x = out_w + kx - 1  inside the halo (offset = tw + kx)
                        accum += shared_input[c_local][th + ky][tw + kx] *
                                 shared_weight[c_local][ky][kx];
                    }
                }
            }
        }

        __syncthreads();
    }

    // 4) Write output
    if (valid) {
        int out_idx = ((n * C_out + out_c) * H + out_h) * W + out_w;
        output[out_idx] = accum;
    }
}

torch::Tensor fused_bn_relu_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_running_mean,
    torch::Tensor bn_running_var,
    torch::Tensor conv_weight,
    float eps)
{
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    const int C_out = conv_weight.size(0);

    auto output = torch::empty({N, C_out, H, W}, input.options());

    dim3 block(TILE_W, TILE_H);  // 16 x 16 threads
    dim3 grid(N * C_out, ((H + TILE_H - 1) / TILE_H) * ((W + TILE_W - 1) / TILE_W));

    fused_bn_relu_conv2d_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        bn_running_mean.data_ptr<float>(),
        bn_running_var.data_ptr<float>(),
        conv_weight.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, H, W, C_out,
        eps
    );

    return output;
}
"""

fused_bn_relu_conv2d_cpp_source = (
    "torch::Tensor fused_bn_relu_conv2d_cuda(torch::Tensor input, torch::Tensor bn_weight, torch::Tensor bn_bias, "
    "torch::Tensor bn_running_mean, torch::Tensor bn_running_var, torch::Tensor conv_weight, float eps);"
)

# Compile the inline CUDA module
_fused_ops = load_inline(
    name="fused_bn_relu_conv2d",
    cpp_sources=fused_bn_relu_conv2d_cpp_source,
    cuda_sources=fused_bn_relu_conv2d_source,
    functions=["fused_bn_relu_conv2d_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# ------------------------------------------------------------
# A drop‑in replacement for the original nn.Sequential
# ------------------------------------------------------------
class FusedDenseLayer(nn.Module):
    def __init__(self, in_features: int, growth_rate: int):
        super(FusedDenseLayer, self).__init__()
        # Keep native modules to own the learnable parameters.
        self.bn = nn.BatchNorm2d(in_features)
        self.conv = nn.Conv2d(in_features, growth_rate, kernel_size=3,
                              padding=1, bias=False)
        # Dropout(0.0) is a no‑op, so we simply omit it.

    def forward(self, x):
        if self.training:
            # Fallback to default PyTorch path for correct training behavior.
            out = self.bn(x)
            out = F.relu(out, inplace=True)
            out = self.conv(out)
        else:
            # Fast fused inference path.
            out = _fused_ops.fused_bn_relu_conv2d_cuda(
                x,
                self.bn.weight,
                self.bn.bias,
                self.bn.running_mean,
                self.bn.running_var,
                self.conv.weight,
                self.bn.eps,
            )
        return out


# ------------------------------------------------------------
# Optimized model
# ------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super(ModelNew, self).__init__()
        layers = []
        for i in range(num_layers):
            in_features = num_input_features + i * growth_rate
            layers.append(FusedDenseLayer(in_features, growth_rate))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(x)
            features.append(new_feature)
            x = torch.cat(features, 1)
        return x