import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import repeat
import collections.abc
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for window partition + spatial MLP + window reverse fusion
spatial_mlp_fused_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void spatial_mlp_fused_kernel(
    const float* __restrict__ input,        // [B, H, W, C]
    float* __restrict__ output,             // [B, H, W, C]
    const float* __restrict__ weight,       // [num_heads*window_size*window_size, num_heads*window_size*window_size] grouped conv weight
    int B, int H, int W, int C,
    int num_heads, int window_size,
    int shift_size,
    const int* __restrict__ padding         // [P_l, P_r, P_t, P_b]
) {
    int total_windows = (H / window_size) * (W / window_size);
    int window_idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    
    if (window_idx >= total_windows * B) return;
    
    int b = window_idx / total_windows;
    int local_idx = window_idx % total_windows;
    int wy = local_idx / (W / window_size);
    int wx = local_idx % (W / window_size);
    
    int C_per_head = C / num_heads;
    int window_area = window_size * window_size;
    int group_size = window_area;
    
    // Shared memory for window data
    extern __shared__ float shared_data[];
    float* window_data = shared_data;  // [window_area, C]
    float* window_out = shared_data + window_area * C;  // [window_area, C]
    
    // Load window data with shift handling
    for (int i = thread_idx; i < window_area * C; i += blockDim.x) {
        int c = i % C;
        int pos = i / C;
        int wy_local = pos / window_size;
        int wx_local = pos % window_size;
        
        int global_h = wy * window_size + wy_local;
        int global_w = wx * window_size + wx_local;
        
        // Apply shift
        if (shift_size > 0) {
            global_h = global_h + padding[2]; // P_t
            global_w = global_w + padding[0]; // P_l
        }
        
        if (global_h >= 0 && global_h < H + (shift_size > 0 ? padding[2] + padding[3] : 0) &&
            global_w >= 0 && global_w < W + (shift_size > 0 ? padding[0] + padding[1] : 0)) {
            int src_h = global_h;
            int src_w = global_w;
            if (shift_size > 0) {
                if (src_h < padding[2] || src_h >= H + padding[2] || 
                    src_w < padding[0] || src_w >= W + padding[0]) {
                    window_data[i] = 0.0f;
                } else {
                    src_h -= padding[2];
                    src_w -= padding[0];
                    window_data[i] = input[((b * H + src_h) * W + src_w) * C + c];
                }
            } else {
                window_data[i] = input[((b * H + global_h) * W + global_w) * C + c];
            }
        } else {
            window_data[i] = 0.0f;
        }
    }
    __syncthreads();
    
    // Perform grouped 1x1 convolution (spatial MLP)
    // weight is [num_heads*window_area, num_heads*window_area] but grouped
    // For each head, it's a linear transform on window_area channels
    for (int h = 0; h < num_heads; h++) {
        int head_offset = h * C_per_head;
        int weight_head_offset = h * window_area * window_area;
        
        for (int pos = thread_idx; pos < window_area; pos += blockDim.x) {
            float sum = 0.0f;
            for (int k = 0; k < window_area; k++) {
                sum += window_data[k * C + head_offset] * weight[weight_head_offset + pos * window_area + k];
            }
            window_out[pos * C + head_offset] = sum;
        }
    }
    __syncthreads();
    
    // Write back with reverse shift
    for (int i = thread_idx; i < window_area * C; i += blockDim.x) {
        int c = i % C;
        int pos = i / C;
        int wy_local = pos / window_size;
        int wx_local = pos % window_size;
        
        int global_h = wy * window_size + wy_local;
        int global_w = wx * window_size + wx_local;
        
        if (shift_size > 0) {
            global_h = global_h + padding[2];
            global_w = global_w + padding[0];
            if (global_h >= padding[2] && global_h < H + padding[2] &&
                global_w >= padding[0] && global_w < W + padding[0]) {
                int dst_h = global_h - padding[2];
                int dst_w = global_w - padding[0];
                output[((b * H + dst_h) * W + dst_w) * C + c] = window_out[i];
            }
        } else {
            if (global_h < H && global_w < W) {
                output[((b * H + global_h) * W + global_w) * C + c] = window_out[i];
            }
        }
    }
}

torch::Tensor spatial_mlp_fused_cuda(
    torch::Tensor input,        // [B, H, W, C]
    torch::Tensor weight,       // [num_heads*window_size*window_size, num_heads*window_size*window_size]
    int num_heads, int window_size, int shift_size,
    torch::Tensor padding       // [4] P_l, P_r, P_t, P_b
) {
    int B = input.size(0);
    int H = input.size(1);
    int W = input.size(2);
    int C = input.size(3);
    
    auto output = torch::zeros_like(input);
    
    int total_windows = (H / window_size) * (W / window_size) * B;
    int threads = 256;
    int shared_mem_size = 2 * window_size * window_size * C * sizeof(float);
    
    spatial_mlp_fused_kernel<<<total_windows, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        weight.data_ptr<float>(),
        B, H, W, C,
        num_heads, window_size,
        shift_size,
        padding.data_ptr<int>()
    );
    
    return output;
}
"""

spatial_mlp_fused_cpp_source = """
torch::Tensor spatial_mlp_fused_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int num_heads, int window_size, int shift_size,
    torch::Tensor padding
);
"""

spatial_mlp_fused = load_inline(
    name="spatial_mlp_fused",
    cpp_sources=spatial_mlp_fused_cpp_source,
    cuda_sources=spatial_mlp_fused_source,
    functions=["spatial_mlp_fused_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Custom CUDA kernel for MLP fusion (Linear + GELU + Dropout + Linear + Dropout)
mlp_fused_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mlp_fused_kernel(
    const float* __restrict__ input,      // [B, L, in_features]
    float* __restrict__ output,           // [B, L, out_features]
    const float* __restrict__ fc1_weight, // [hidden_features, in_features]
    const float* __restrict__ fc1_bias,   // [hidden_features]
    const float* __restrict__ fc2_weight, // [out_features, hidden_features]
    const float* __restrict__ fc2_bias,   // [out_features]
    int B, int L, int in_features, int hidden_features, int out_features,
    float dropout_prob
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * L * out_features;
    
    if (idx >= total_elements) return;
    
    int out_idx = idx % out_features;
    int seq_idx = (idx / out_features) % L;
    int batch_idx = idx / (L * out_features);
    
    // First linear + GELU
    float hidden_val = fc1_bias[out_idx % hidden_features];
    int hidden_idx = out_idx % hidden_features;
    for (int i = 0; i < in_features; i++) {
        hidden_val += input[(batch_idx * L + seq_idx) * in_features + i] * 
                      fc1_weight[hidden_idx * in_features + i];
    }
    
    // GELU activation
    float x = hidden_val;
    float gelu = x * 0.5f * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
    
    // Dropout (simplified: scale during training, here we just pass through)
    // In practice, dropout would use random mask, but for inference we skip it
    // or use scaling. Here we assume eval mode (no dropout) for simplicity.
    // For training, you'd need to generate random numbers.
    float dropped = gelu; // In eval mode, no dropout
    
    // Second linear
    float out_val = fc2_bias[out_idx];
    for (int i = 0; i < hidden_features; i++) {
        out_val += dropped * fc2_weight[out_idx * hidden_features + i];
    }
    
    // Second dropout (eval mode, no dropout)
    output[idx] = out_val;
}

torch::Tensor mlp_fused_cuda(
    torch::Tensor input,
    torch::Tensor fc1_weight, torch::Tensor fc1_bias,
    torch::Tensor fc2_weight, torch::Tensor fc2_bias,
    int hidden_features, int out_features,
    float dropout_prob
) {
    int B = input.size(0);
    int L = input.size(1);
    int in_features = input.size(2);
    
    auto output = torch::zeros({B, L, out_features}, input.options());
    
    int total_elements = B * L * out_features;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    mlp_fused_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        fc1_weight.data_ptr<float>(),
        fc1_bias.data_ptr<float>(),
        fc2_weight.data_ptr<float>(),
        fc2_bias.data_ptr<float>(),
        B, L, in_features, hidden_features, out_features,
        dropout_prob
    );
    
    return output;
}
"""

mlp_fused_cpp_source = """
torch::Tensor mlp_fused_cuda(
    torch::Tensor input,
    torch::Tensor fc1_weight, torch::Tensor fc1_bias,
    torch::Tensor fc2_weight, torch::Tensor fc2_bias,
    int hidden_features, int out_features,
    float dropout_prob
);
"""

mlp_fused = load_inline(
    name="mlp_fused",
    cpp_sources=mlp_fused_cpp_source,
    cuda_sources=mlp_fused_source,
    functions=["mlp_fused_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Custom CUDA kernel for patch merging
patch_merging_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void patch_merging_kernel(
    const float* __restrict__ input,      // [B, H, W, C]
    float* __restrict__ output,           // [B, H/2, W/2, 2*C]
    const float* __restrict__ norm_weight, // [4*C]
    const float* __restrict__ norm_bias,   // [4*C]
    const float* __restrict__ reduction_weight, // [2*C, 4*C]
    int B, int H, int W, int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_H = H / 2;
    int out_W = W / 2;
    int out_C = 2 * C;
    int total_elements = B * out_H * out_W * out_C;
    
    if (idx >= total_elements) return;
    
    int c_out = idx % out_C;
    int w_out = (idx / out_C) % out_W;
    int h_out = (idx / (out_C * out_W)) % out_H;
    int b = idx / (out_C * out_W * out_H);
    
    // Gather 4 patches
    float x0 = input[((b * H + h_out * 2) * W + w_out * 2) * C + (c_out % C)];
    float x1 = input[((b * H + h_out * 2 + 1) * W + w_out * 2) * C + (c_out % C)];
    float x2 = input[((b * H + h_out * 2) * W + w_out * 2 + 1) * C + (c_out % C)];
    float x3 = input[((b * H + h_out * 2 + 1) * W + w_out * 2 + 1) * C + (c_out % C)];
    
    // Concatenate
    float concat[4];
    concat[0] = x0;
    concat[1] = x1;
    concat[2] = x2;
    concat[3] = x3;
    
    // LayerNorm
    float mean = 0.0f;
    for (int i = 0; i < 4; i++) {
        mean += concat[i];
    }
    mean /= 4.0f;
    
    float var = 0.0f;
    for (int i = 0; i < 4; i++) {
        float diff = concat[i] - mean;
        var += diff * diff;
    }
    var = sqrtf(var / 4.0f + 1e-5f);
    
    float normalized[4];
    for (int i = 0; i < 4; i++) {
        normalized[i] = (concat[i] - mean) / var * norm_weight[i] + norm_bias[i];
    }
    
    // Linear reduction
    float sum = 0.0f;
    for (int i = 0; i < 4; i++) {
        sum += normalized[i] * reduction_weight[c_out * 4 + i];
    }
    
    output[idx] = sum;
}

torch::Tensor patch_merging_cuda(
    torch::Tensor input,
    torch::Tensor norm_weight, torch::Tensor norm_bias,
    torch::Tensor reduction_weight
) {
    int B = input.size(0);
    int H = input.size(1);
    int W = input.size(2);
    int C = input.size(3);
    
    int out_H = H / 2;
    int out_W = W / 2;
    int out_C = 2 * C;
    
    auto output = torch::zeros({B, out_H, out_W, out_C}, input.options());
    
    int total_elements = B * out_H * out_W * out_C;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    patch_merging_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        norm_weight.data_ptr<float>(),
        norm_bias.data_ptr<float>(),
        reduction_weight.data_ptr<float>(),
        B, H, W, C
    );
    
    return output;
}
"""

patch_merging_cpp_source = """
torch::Tensor patch_merging_cuda(
    torch::Tensor input,
    torch::Tensor norm_weight, torch::Tensor norm_bias,
    torch::Tensor reduction_weight
);
"""

patch_merging = load_inline(
    name="patch_merging",
    cpp_sources=patch_merging_cpp_source,
    cuda_sources=patch_merging_source,
    functions=["patch_merging_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Original helper functions (unchanged)
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.drop_prob = drop

    def forward(self, x):
        return mlp_fused.mlp_fused_cuda(
            x,
            self.fc1.weight, self.fc1.bias,
            self.fc2.weight, self.fc2.bias,
            self.hidden_features, self.out_features,
            self.drop_prob
        )

class SwinMLPBlock(nn.Module):
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

        self.norm1 = norm_layer(dim)
        self.spatial_mlp = nn.Conv1d(self.num_heads * self.window_size ** 2,
                                     self.num_heads * self.window_size ** 2,
                                     kernel_size=1,
                                     groups=self.num_heads)

        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Use fused CUDA kernel for window partition, spatial MLP, and window reverse
        padding_tensor = torch.tensor(self.padding, dtype=torch.int32, device=x.device)
        x = spatial_mlp_fused.spatial_mlp_fused_cuda(
            x,
            self.spatial_mlp.weight.squeeze(-1),  # Conv1d weight is [out, in, 1], we need [out, in]
            self.num_heads,
            self.window_size,
            self.shift_size,
            padding_tensor
        )
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
        x = patch_merging.patch_merging_cuda(
            x,
            self.norm.weight,
            self.norm.bias,
            self.reduction.weight
        )
        x = x.view(B, -1, 2 * self.dim)
        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinMLPBlock(dim=dim, input_resolution=input_resolution,
                         num_heads=num_heads, window_size=window_size,
                         shift_size=0 if (i % 2 == 0) else window_size // 2,
                         mlp_ratio=mlp_ratio,
                         drop=drop,
                         drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                         norm_layer=norm_layer)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

class ModelNew(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               drop=drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x