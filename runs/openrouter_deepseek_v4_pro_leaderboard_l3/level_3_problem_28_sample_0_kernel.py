import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA source for fused patch embedding
patch_embed_cpp_source = "torch::Tensor patch_embed_cuda(torch::Tensor img, torch::Tensor weight, torch::Tensor bias, torch::Tensor cls_token, torch::Tensor pos_embed, int patch_size);"

patch_embed_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void patch_embed_kernel(
    const float* __restrict__ img,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ cls_token,
    const float* __restrict__ pos_embed,
    float* __restrict__ out,
    int B, int C, int H, int W,
    int patch_size,
    int num_patches_h, int num_patches_w,
    int dim,
    int patch_dim
) {
    int S = num_patches_h * num_patches_w + 1;
    int seq_idx = blockIdx.x % S;
    int b = blockIdx.x / S;
    int d = threadIdx.x;

    if (d >= dim) return;

    int out_idx = b * S * dim + seq_idx * dim + d;

    if (seq_idx == 0) {
        // CLS token position: simply add cls_token and pos_embedding
        out[out_idx] = cls_token[d] + pos_embed[seq_idx * dim + d];
        return;
    }

    int patch_idx = seq_idx - 1;
    int ph = patch_idx / num_patches_w;
    int pw = patch_idx % num_patches_w;

    // Load the image patch into shared memory
    extern __shared__ float patch_vals[];
    for (int i = threadIdx.x; i < patch_dim; i += blockDim.x) {
        int c = i / (patch_size * patch_size);
        int residual = i % (patch_size * patch_size);
        int pi = residual / patch_size;
        int pj = residual % patch_size;
        int h_idx = ph * patch_size + pi;
        int w_idx = pw * patch_size + pj;
        patch_vals[i] = img[b * C * H * W + c * H * W + h_idx * W + w_idx];
    }
    __syncthreads();

    // Dot product with weight row + bias, then add positional embedding
    float sum = bias[d];
    for (int k = 0; k < patch_dim; k++) {
        sum += patch_vals[k] * weight[d * patch_dim + k];
    }
    sum += pos_embed[seq_idx * dim + d];
    out[out_idx] = sum;
}

torch::Tensor patch_embed_cuda(
    torch::Tensor img,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor cls_token,
    torch::Tensor pos_embed,
    int patch_size
) {
    const auto B = img.size(0);
    const auto C = img.size(1);
    const auto H = img.size(2);
    const auto W = img.size(3);
    const auto dim = weight.size(0);
    const auto patch_dim = weight.size(1);
    const int num_patches_h = H / patch_size;
    const int num_patches_w = W / patch_size;
    const int S = num_patches_h * num_patches_w + 1;

    auto out = torch::empty({B, S, dim}, img.options());

    const int threads = std::min(dim, 1024);
    const int blocks = B * S;

    patch_embed_kernel<<<blocks, threads, patch_dim * sizeof(float)>>>(
        img.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        cls_token.data_ptr<float>(),
        pos_embed.data_ptr<float>(),
        out.data_ptr<float>(),
        B, C, H, W,
        patch_size,
        num_patches_h, num_patches_w,
        dim,
        patch_dim
    );

    return out;
}
"""

# Compile the inline CUDA module
patch_embed_cuda_op = load_inline(
    name="patch_embed_cuda_op",
    cpp_sources=patch_embed_cpp_source,
    cuda_sources=patch_embed_cuda_source,
    functions=["patch_embed_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout=0.1, emb_dropout=0.1):
        super(ModelNew, self).__init__()
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)   # kept for weight/bias, not used directly
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout),
            num_layers=depth
        )

        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, img):
        # Replaces: unfold + reshape + Linear + cls token cat + positional add
        x = patch_embed_cuda_op.patch_embed_cuda(
            img,
            self.patch_to_embedding.weight,
            self.patch_to_embedding.bias,
            self.cls_token,
            self.pos_embedding,
            self.patch_size
        )
        x = self.dropout(x)          # dropout (identity when rate=0)
        x = self.transformer(x)      # (batch, seq, dim) – kept as original
        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)