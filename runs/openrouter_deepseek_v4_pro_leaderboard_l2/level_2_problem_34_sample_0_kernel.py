import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Fused LayerNorm + GELU + scaling CUDA kernel source
fused_op_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_layernorm_gelu_scale_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int N, int C, int D, int H, int W,
    float eps, float scale) {

    int total_slices = N * C * D * H;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_slices) return;

    // base offset for this slice: each slice occupies W elements
    const float* slice_in = x + idx * W;
    float* slice_out = out + idx * W;

    // Compute mean
    float sum = 0.0f;
    for (int i = 0; i < W; ++i) {
        sum += slice_in[i];
    }
    float mean = sum / W;

    // Compute variance
    float sq_sum = 0.0f;
    for (int i = 0; i < W; ++i) {
        float diff = slice_in[i] - mean;
        sq_sum += diff * diff;
    }
    float var = sq_sum / W;

    float inv_std = rsqrtf(var + eps);

    // Apply normalization, GELU, and scaling
    for (int i = 0; i < W; ++i) {
        float val = (slice_in[i] - mean) * inv_std;
        // GELU: 0.5 * val * (1 + erf(val / sqrt(2)))
        float gelu_val = 0.5f * val * (1.0f + erff(val * 0.7071067811865475f)); // 1/sqrt(2)
        slice_out[i] = gelu_val * scale;
    }
}

torch::Tensor fused_layernorm_gelu_scale_cuda(
    torch::Tensor x,
    float eps,
    float scale) {
    
    TORCH_CHECK(x.dim() == 5, "Input must be 5D (N,C,D,H,W)");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    
    int N = x.size(0);
    int C = x.size(1);
    int D = x.size(2);
    int H = x.size(3);
    int W = x.size(4);
    
    auto out = torch::empty_like(x);
    
    int total_slices = N * C * D * H;
    const int threads = 256;
    const int blocks = (total_slices + threads - 1) / threads;
    
    fused_layernorm_gelu_scale_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, D, H, W,
        eps, scale);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\\n", cudaGetErrorString(err));
    }
    
    return out;
}
"""

fused_op_cpp_source = """
torch::Tensor fused_layernorm_gelu_scale_cuda(
    torch::Tensor x,
    float eps,
    float scale);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_layernorm_gelu_scale",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_cuda_source,
    functions=["fused_layernorm_gelu_scale_cuda"],
    verbose=True,
    extra_cflags=[],
    extra_ldflags=[],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias
        )
        self.eps = eps
        self.scaling_factor = scaling_factor
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x.contiguous()  # Ensure contiguous for kernel
        x = self.fused_op.fused_layernorm_gelu_scale_cuda(x, self.eps, self.scaling_factor)
        return x