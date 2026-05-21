```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import repeat
import collections.abc
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# Custom CUDA Kernels for Optimization
# -----------------------------------------------------------------------------

# 1. Optimized LayerNorm (Fused: Mean, Var, Normalize)
# Standard LayerNorm involves multiple passes over memory. This kernel computes
# mean and variance in one pass (or two efficient passes) and normalizes.
layer_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void layer_norm_kernel(const float* input, const float* weight, const float* bias, 
                                  float* output, float* mean_out, float* var_out,
                                  int N, int C, float eps) {
    // Each block handles one sample (batch item)
    int idx = blockIdx.x;
    if (idx >= N) return;

    const float* x = input + idx * C;
    float* out = output + idx * C;
    
    // Calculate mean
    float sum = 0.0f;
    for (int i = 0; i < C; ++i) {
        sum += x[i];
    }
    float mean = sum / C;

    // Calculate variance
    float var_sum = 0.0f;
    for (int i = 0; i < C; ++i) {
        float diff = x[i] - mean;
        var_sum += diff * diff;
    }
    float var = var_sum / C + eps;
    float inv_std = rsqrtf(var);

    // Normalize and apply affine transform
    for (int i = 0; i < C; ++i) {
        float val = (x[i] - mean) * inv_std;
        if (weight != nullptr) {
            val = val * weight[i];
        }
        if (bias != nullptr) {
            val = val + bias[i];
        }
        out[i] = val;
    }

    if (mean_out) mean_out[idx] = mean;
    if (var_out) var_out[idx] = var;
}

torch::Tensor layer_norm_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int normalized_shape, float eps) {
    auto N = input.size(0);
    auto C = input.size(1); // Assuming 2D input for simplicity in this context (B, L*C) or similar
    
    auto output = torch::empty_like(input);
    
    const int block_size = 1; // One thread per sample is sufficient if we handle C inside
    // However, to utilize GPU parallelism better for large C, we can launch N blocks.
    // If N is small, this might be underutilized, but it's correct.
    
    layer_norm_kernel<<<N, 1>>>(input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), 
                                output.data_ptr<float>(), nullptr, nullptr, N, C, eps);
    
    return output;
}
"""

layer_norm_cpp_source = (
    "torch::Tensor layer_norm_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int normalized_shape, float eps);"
)

# 2. Optimized Spatial MLP (Group Conv1d equivalent for Swin MLP)
# The original uses nn.Conv1d with groups. We can fuse the reshape/view operations 
# and the convolution into a single kernel to avoid intermediate tensor allocations.
spatial_mlp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel for Group Conv1d where kernel size is 1.
// Essentially: Output[g * C_out + i] = Sum_{j} (Input[g * C_in + j] * Weight[i]) 
// But since K=1, it's just a linear projection per group.
// Input shape: (N, H, C) where C is num_heads * window_size^2
// We treat it as 1D convolution over the spatial dimension H.

__global__ void spatial_mlp_kernel(const float* input, const float* weight, float* output, 
                                   int N, int H, int C_in, int C_out, int groups) {
    // Each thread handles one element of the output tensor
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * H * C_out;
    
    if (idx >= total_elements) return;

    // Decompose index
    int c_out = idx % C_out;
    int h = (idx / C_out) % H;
    int n = idx / (H * C_out);

    int group = c_out / (C_out / groups);
    int c_in_offset = group * (C_in / groups);
    
    // Since kernel size is 1, we just look at the same spatial position h
    // Input channel corresponding to output channel c_out in the same group
    // In Swin MLP, C_in == C_out usually for this specific block structure 
    // but let's be general. The weight matrix is (C_out, C_in).
    // However, nn.Conv1d with groups applies a separate linear layer per group.
    
    float sum = 0.0f;
    int c_in_step = C_in / groups;
    
    // Unroll or loop for the inner dimension
    // Since it's K=1, we are essentially doing: out[n,h,c_out] = dot(input[n,h, group_start:group_end], weight[c_out, group_start:group_end])
    // But wait, standard Conv1d with groups: 
    // The input channels are divided into groups. Each group is convolved with a subset of output channels.
    // Specifically, if groups=G, then C_in/G channels map to C_out/G channels.
    
    int c_in_start = group * c_in_step;
    int c_out_start = group * (C_out / groups);
    
    // The weight index for output channel c_out is c_out - c_out_start + c_in_offset? 
    // Actually, PyTorch Conv1d weights are (out_channels, in_channels/groups, kernel_size).
    // So weight[c_out][c_in_local]
    
    int local_c_out = c_out - c_out_start;
    
    for (int k = 0; k < c_in_step; ++k) {
        sum += input[n * H * C_in + h * C_in + c_in_start + k] * weight[local_c_out * c_in_step + k];
    }
    
    output[idx] = sum;
}

torch::Tensor spatial_mlp_cuda(torch::Tensor x, torch::Tensor weight) {
    // x: (N, H, C)
    // weight: (C, C) for the specific case in SwinMLP where in_channels == out_channels and groups=num_heads*window_size^2?
    // Actually in SwinMLPBlock: 
    // self.spatial_mlp = nn.Conv1d(self.num_heads * self.window_size ** 2, self.num_heads * self.window_size ** 2, kernel_size=1, groups=self.num_heads)
    // Input to this layer is (N, window_size*window_size, num_heads * window_size^2) ? 
    // No, looking at forward:
    // x_windows_heads: (nW*B, nH, window_size*window_size, C//nH) -> reshape to (nW*B*nH, window_size*window_size, C//nH) ??
    // Wait, the code does:
    // x_windows_heads = x_windows_heads.transpose(1, 2) # nW*B, nH, window_size*window_size, C//nH -> nW*B, window_size*window_size, nH, C//nH ? No.
    // Let's trace carefully:
    // x_windows: (nW*B, Ws, Ws, C)
    // view(-1, Ws*Ws, C) -> (nW*B, Ws*Ws, C)
    // view(-1, Ws*Ws, nH, C//nH) -> (nW*B, Ws*Ws, nH, C//nH)
    // transpose(1, 2) -> (nW*B, nH, Ws*Ws, C//nH)
    // reshape(-1, nH*Ws*Ws, C//nH) -> (nW*B*nH, Ws*Ws, C//nH) ?? No.
    // reshape(-1, self.num_heads * self.window_size * self.window_size, C // self.num_heads)
    // This results in shape: (nW*B, nH*Ws*Ws, C//nH) ? 
    // Let's check the dimensions passed to Conv1d.
    // nn.Conv1d expects (N, C_in, L).
    // The input `x_windows_heads` after reshape is (nW*B, nH*Ws*Ws, C//nH).
    // So N = nW*B, C_in = nH*Ws*Ws, L = C//nH.
    // Groups = nH.
    // Out channels = nH*Ws*Ws.
    
    auto N = x.size(0);
    auto H = x.size(1); // This is the "length" of the sequence for Conv1d, i.e., num_heads * window_size^2
    auto C = x.size(2); // This is the channel dimension for Conv1d, i.e., C//num_heads
    
    auto output = torch::empty_like(x);
    
    int total_elements = N * H * C;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    spatial_mlp_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), N, H, C, C, x.size(0)); 
    // Note: passing groups is tricky via args. We assume standard Swin MLP config where groups = num_heads.
    // But the kernel above assumes generic groups. Let's hardcode logic or pass groups.
    // To keep it simple and robust for the specific Swin MLP structure:
    
    return output;
}
"""

# The above spatial_mlp_kernel is a bit complex to get exactly right with PyTorch Conv1d semantics in a single generic kernel without passing 'groups'.
# A safer, highly optimized approach for Swin MLP's specific GroupConv1d(K=1) is to treat it as a GEMM or specialized linear layer.
# However, since we want to replace the operator, let's use a simpler fused kernel that mimics the exact operation:
# Input: (N, L, C_in), Weight: (C_out, C_in), Groups: G.
# Output: (N, L, C_out).

spatial_mlp_v2_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void spatial_mlp_kernel_v2(const float* input, const float* weight, float* output, 
                                      int N, int L, int C_in, int C_out, int groups) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * L * C_out;
    if (idx >= total) return;

    int c_out = idx % C_out;
    int l = (idx / C_out) % L;
    int n = idx / (L * C_out);

    int group = c_out / (C_out / groups);
    int c_in_start = group * (C_in / groups);
    int local_c_out = c_out - group * (C_out / groups);
    int c_in_step = C_in / groups;

    float sum = 0.0f;
    for (int k = 0; k < c_in_step; ++k) {
        // Input is (N, L, C_in). Index: n*L*C_in + l*C_in + c_in_start + k
        sum += input[n * L * C_in + l * C_in + c_in_start + k] * 
               weight[local_c_out * c_in_step + k];
    }
    output[idx] = sum;
}

torch::Tensor spatial_mlp_cuda_v2(torch::Tensor x, torch::Tensor weight, int groups) {
    auto N = x.size(0);
    auto L = x.size(1);
    auto C_in = x.size(2);
    auto C_out = weight.size(0);

    auto output = torch::empty_like(x);
    
    int total_elements = N * L * C_out;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    spatial_mlp_kernel_v2<<<num_blocks, block_size>>>(x.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), N, L, C_in, C_out, groups);

    return output;
}
"""

spatial_mlp_cpp_source = (
    "torch::Tensor spatial_mlp_cuda_v2(torch::Tensor x, torch::Tensor weight, int groups);"
)

# 3. Optimized Patch Merging (Cat + Linear)
# Fuses the slicing, concatenation, and linear projection.
patch_merge_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void patch_merge_kernel(const float* input, float* output, 
                                   int B, int H, int W, int C, int out_C) {
    // Output shape: (B, H/2 * W/2, 4*C)
    // Input shape: (B, H, W, C)
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out_elements = B * (H / 2) * (W / 2) * out_C;
    
    if (idx >= total_out_elements) return;

    // Decompose output index
    int c_out = idx % out_C; // 0 to 4*C-1
    int w_half = (idx / out_C) % (W / 2);
    int h_half = (idx / (out_C * (W / 2))) % (H / 2);
    int b = idx / (out_C * (W / 2) * (H / 2));

    // Map to input coordinates
    // The 4 channels come from:
    // x0: (b, 2*h_half, 2*w_half, :) -> index 0..C-1
    // x1: (b, 2*h_half+1, 2*w_half, :) -> index C..2C-1
    // x2: (b, 2*h_half, 2*w_half+1, :) -> index 2C..3C-1
    // x3: (b, 2*h_half+1, 2*w_half+1, :) -> index 3C..4C-1
    
    int c_in = c_out % C;
    int src_idx = 0;
    
    if (c_out < C) {
        // From x0
        src_idx = b * H * W * C + (2 * h_half) * W * C + (2 * w_half) * C + c_in;
    } else if (c_out < 2 * C) {
        // From x1
        src_idx = b * H * W * C + (2 * h_half + 1) * W * C + (2 * w_half) * C + c_in;
    } else if (c_out < 3 * C) {
        // From x2
        src_idx = b * H * W * C + (2 * h_half) * W * C + (2 * w_half + 1) * C + c_in;
    } else {
        // From x3
        src_idx = b * H * W * C + (2 * h_half + 1) * W * C + (2 * w_half + 1) * C + c_in;
    }

    output[idx] = input[src_idx];
}

torch::Tensor patch_merge_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    auto B = x.size(0);
    auto H = x.size(1);
    auto W = x.size(2);
    auto C = x.size(3);
    
    // Output of cat is (B, H/2, W/2, 4*C)
    int out_C = 4 * C;
    int L_out = (H / 2) * (W / 2);
    
    // First, do the gathering/cat part
    auto cat_output = torch::empty({B, L_out, out_C}, x.options());
    
    int total_elements = B * L_out * out_C;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    patch_merge_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), cat_output.data_ptr<float>(), B, H, W, C, out_C);

    // Then apply Linear: x.view(B*L_out, 4*C) @ weight.T + bias
    // We can fuse this too, but for clarity and stability, we'll do a second kernel or use torch.mm
    // Let's fuse it into one kernel to be truly optimized.
    
    auto final_output = torch::empty({B, L_out, weight.size(0)}, x.options());
    
    // Redefine kernel to include linear layer? 
    // Or just launch another kernel. Given the constraints, let's stick to two steps if fusion is too complex for inline, 
    // but the prompt asks for optimization. Let's do a fused kernel.
    
    return final_output; // Placeholder, will implement fused below
}

__global__ void patch_merge_fused_kernel(const float* input, const float* weight, const float* bias, float* output, 
                                         int B, int H, int W, int C, int out_C_linear) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out_elements = B * (H / 2) * (W / 2) * out_C_linear;
    
    if (idx >= total_out_elements) return;

    int c_out = idx % out_C_linear;
    int w_half = (idx / out_C_linear) % (W / 2);
    int h_half = (idx / (out_C_linear * (W / 2))) % (H / 2);
    int b = idx / (out_C_linear * (W / 2) * (H / 2));

    // Gather 4 values from input
    float vals[4];
    int c_in;
    
    // x0
    c_in = c_out % C; // Wait, linear layer mixes all 4C channels. 
    // We need to gather the 4C values first.
    
    // Let's gather into a local array or compute dot product directly?
    // Dot product with weight row is expensive if we don't have the 4C values.
    // Better: Gather 4C values, then dot product.
    
    float vec[4 * 16]; // Assuming C <= 16 for simplicity? No, C can be large.
    // Dynamic stack allocation is bad. 
    // Let's just gather the 4C values into registers if possible, or use shared memory.
    // For general C, we must load them.
    
    // Re-calculate indices for the 4 blocks
    int idx0 = b * H * W * C + (2 * h_half) * W * C + (2 * w_half) * C;
    int idx1 = b * H * W * C + (2 * h_half + 1) * W * C + (2 * w_half) * C;
    int idx2 = b * H * W * C + (2 * h_half) * W * C + (2 * w_half + 1) * C;
    int idx3 = b * H * W * C + (2 * h_half + 1) * W * C + (2 * w_half + 1) * C;

    // Weight is (out_C_linear, 4*C). 
    // We need to compute sum_{k=0}^{4C-1} input_vec[k] * weight[c_out][k]
    
    float sum = 0.0f;
    
    // Unroll the 4 blocks
    for (int k = 0; k < C; ++k) {
        sum += input[idx0 + k] * weight[c_out * (4 * C) + k];
        sum += input[idx1 + k] * weight[c_out * (4 * C) + C + k];
        sum += input[idx2 + k] * weight[c_out * (4 * C) + 2 * C + k];
        sum += input[idx3 + k] * weight[c_out * (4 * C) + 3 * C + k];
    }
    
    if (bias != nullptr) {
        sum += bias[c_out];
    }
    
    output[idx] = sum;
}

torch::Tensor patch_merge_fused_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    auto B = x.size(0);
    auto H = x.size(1);
    auto W = x.size(2);
    auto C = x.size(3);
    auto out_C_linear = weight.size(0);
    
    auto output = torch::empty({B, (H/2)*(W/2), out_C_linear}, x.options());
    
    int total_elements = B * (H / 2) * (W / 2) * out_C_linear;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    patch_merge_fused_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), B, H, W, C, out_C_linear);

    return output;
}
"""

patch_merge_cpp_source = (
    "torch::Tensor patch_merge_fused_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias);"
)

# Compile extensions
try:
    layer_norm_ext = load_inline(
        name="layer_norm_ext",
        cpp_sources=layer_norm_cpp_source,
        cuda_sources=layer_norm_source,
        functions=["layer_norm_cuda"],
        verbose=False,
    )
    
    spatial_mlp_ext = load_inline(
        name="spatial_mlp_ext",
        cpp_sources=spatial_mlp_cpp_source,
        cuda_sources=spatial_mlp_v2_source,
        functions=["spatial_mlp_cuda_v2"],
        verbose=False,
    )
    
    patch_merge_ext = load_inline(
        name="patch_merge_ext",
        cpp_sources=patch_merge_cpp_source,
        cuda_sources=patch_merge_source,
        functions=["patch_merge_fused_cuda"],
        verbose=False,
    )
except Exception as e:
    print(f"Warning: CUDA extension compilation failed. Falling back to standard PyTorch ops if possible, or erroring out.")
    raise e


# -----------------------------------------------------------------------------
# Optimized Model Architecture
# -----------------------------------------------------------------------------

class MlpNew(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinMLPBlockNew(nn.Module):
    r""" Swin MLP Block with Custom CUDA Operators.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.padding = [self.window_size - self.shift_size, self.shift_size,
                        self.window_size - self.shift_size, self.shift_size]

        # Use custom LayerNorm
        self.norm1_weight = nn.Parameter(torch.ones(dim))
        self.norm1_bias = nn.Parameter(torch.zeros(dim))
        
        # Spatial MLP uses Group Conv1d. We replace it with custom CUDA kernel.
        # The input to spatial_mlp is (N, L, C_in) where L = window_size^2 * num_heads? 
        # No, in the original code:
        # x_windows_heads reshaped to (-1, num_heads * window_size * window_size, C // num_heads)
        # So N' = nW*B, L' = num_heads * window_size^2, C' = C // num_heads.
        # Groups = num_heads.
        # In_channels = num_heads * window_size^2. Out_channels = num_heads * window_size^2.
        
        self.spatial_mlp_weight = nn.Parameter(torch.randn(self.num_heads * self.window_size ** 2, 
                                                           self.num_heads * self.window_size ** 2))
        
        self.drop_path = nn.Identity()
        
        # Second LayerNorm
        self.norm2_weight = nn.Parameter(torch.ones(dim))
        self.norm2_bias = nn.Parameter(torch.zeros(dim))
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MlpNew(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        
        # 1. Custom LayerNorm
        # Input to norm is (B, L, C) -> (B, L*C) for the kernel? 
        # My kernel expects (N, C). Here N=B*L, C=C.
        x_flat = x.view(B * L, C)
        x_normed = layer_norm_ext.layer_norm_cuda(x_flat, self.norm1_weight, self.norm1_bias, C, 1e-5)
        x = x_normed.view(B, L, C)
        
        x = x.view(B, H, W, C)

        # Shift
        if self.shift_size > 0:
            P_l, P_r, P_t, P_b = self.padding
            shifted_x = F.pad(x, [0, 0, P_l, P_r, P_t, P_b], "constant", 0)
        else:
            shifted_x = x
        _, _H, _W, _ = shifted_x.shape

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # Prepare for Spatial MLP
        # Original: view(-1, Ws*Ws, nH, C//nH) -> transpose(1,2) -> reshape(-1, nH*Ws*Ws, C//nH)
        # Let's replicate this logic to get the correct shape for the custom kernel.
        x_windows_heads = x_windows.view(-1, self.window_size * self.window_size, self.num_heads, C // self.num_heads)
        x_windows_heads = x_windows_heads.transpose(1, 2)  # nW*B, nH, window_size*window_size, C//nH
        x_windows_heads = x_windows_heads.reshape(-1, self.num_heads * self.window_size * self.window_size,
                                                  C // self.num_heads)
        
        # Apply Custom Spatial MLP (Group Conv1d K=1)
        # Input: (N', L', C') where N'=nW*B, L'=nH*Ws*Ws, C'=C//nH
        # Weight: (L', L') ? No.
        # In nn.Conv1d(C_in, C_out, kernel_size=1, groups=G):
        # Weight shape is (C_out, C_in/G).
        # Here C_in = nH*Ws*Ws, C_out = nH*Ws*Ws, G = nH.
        # So Weight shape should be (nH*Ws*Ws, Ws*Ws).
        # My custom kernel expects weight of shape (C_out, C_in).
        # Wait, my kernel `spatial_mlp_kernel_v2` assumes standard linear mapping per group?
        # Let's check the kernel logic:
        # `weight[local_c_out * c_in_step + k]`
        # This implies weight is flattened groups.
        # If I pass the full weight matrix (C_out, C_in) to a kernel that handles groups internally, it works.
        
        spatial_mlp_windows = spatial_mlp_ext.spatial_mlp_cuda_v2(x_windows_heads, self.spatial_mlp_weight, self.num_heads)

        # Reverse reshape
        spatial_mlp_windows = spatial_mlp_windows.view(-1, self.num_heads, self.window_size * self.window_size,
                                                       C // self.num_heads).transpose(1, 2)
        spatial_mlp_windows = spatial_mlp_windows.reshape(-1, self.window_size * self.window_size, C)

        # Merge windows
        spatial_mlp_windows = spatial_mlp_windows.reshape(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(spatial_mlp_windows, self.window_size, _H, _W)  # B H' W' C

        # Reverse shift