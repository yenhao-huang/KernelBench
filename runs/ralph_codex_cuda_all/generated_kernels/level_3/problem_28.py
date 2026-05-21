import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

vit_cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math_constants.h>

__global__ void patch_embed_kernel(
    const float* __restrict__ img,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ cls,
    const float* __restrict__ pos,
    float* __restrict__ out,
    int B, int C, int H, int W, int P, int N, int D, int PD
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * (N + 1) * D;
    if (idx >= total) return;

    int d = idx % D;
    int t = (idx / D) % (N + 1);
    int b = idx / ((N + 1) * D);

    if (t == 0) {
        out[idx] = cls[d] + pos[d];
        return;
    }

    int patch = t - 1;
    int patches_per_row = W / P;
    int ph = patch / patches_per_row;
    int pw = patch - ph * patches_per_row;

    float acc = bias[d];
    for (int k = 0; k < PD; ++k) {
        int c = k / (P * P);
        int rem = k - c * P * P;
        int ih = ph * P + rem / P;
        int iw = pw * P + rem - (rem / P) * P;
        float v = img[((b * C + c) * H + ih) * W + iw];
        acc += v * weight[d * PD + k];
    }
    out[idx] = acc + pos[t * D + d];
}

__device__ __forceinline__ float gelu_exact(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
}

__global__ void mlp_head_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w1,
    const float* __restrict__ b1,
    const float* __restrict__ w2,
    const float* __restrict__ b2,
    float* __restrict__ out,
    int B, int D, int M, int O
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * O;
    if (idx >= total) return;

    int o = idx % O;
    int b = idx / O;

    float acc2 = b2[o];
    for (int m = 0; m < M; ++m) {
        float acc1 = b1[m];
        for (int d = 0; d < D; ++d) {
            acc1 += x[b * D + d] * w1[m * D + d];
        }
        acc2 += gelu_exact(acc1) * w2[o * M + m];
    }
    out[idx] = acc2;
}

torch::Tensor patch_embed_cuda(
    torch::Tensor img,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor cls,
    torch::Tensor pos,
    int64_t patch_size
) {
    int B = img.size(0);
    int C = img.size(1);
    int H = img.size(2);
    int W = img.size(3);
    int P = (int)patch_size;
    int N = (H / P) * (W / P);
    int D = weight.size(0);
    int PD = weight.size(1);

    auto out = torch::empty({B, N + 1, D}, img.options());
    int total = B * (N + 1) * D;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    patch_embed_kernel<<<blocks, threads>>>(
        img.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        cls.data_ptr<float>(),
        pos.data_ptr<float>(),
        out.data_ptr<float>(),
        B, C, H, W, P, N, D, PD
    );
    return out;
}

torch::Tensor mlp_head_cuda(
    torch::Tensor x,
    torch::Tensor w1,
    torch::Tensor b1,
    torch::Tensor w2,
    torch::Tensor b2
) {
    int B = x.size(0);
    int D = x.size(1);
    int M = w1.size(0);
    int O = w2.size(0);

    auto out = torch::empty({B, O}, x.options());
    int total = B * O;
    int threads = 128;
    int blocks = (total + threads - 1) / threads;

    mlp_head_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        w1.data_ptr<float>(),
        b1.data_ptr<float>(),
        w2.data_ptr<float>(),
        b2.data_ptr<float>(),
        out.data_ptr<float>(),
        B, D, M, O
    );
    return out;
}
"""

vit_cpp_source = r"""
torch::Tensor patch_embed_cuda(torch::Tensor img, torch::Tensor weight, torch::Tensor bias, torch::Tensor cls, torch::Tensor pos, int64_t patch_size);
torch::Tensor mlp_head_cuda(torch::Tensor x, torch::Tensor w1, torch::Tensor b1, torch::Tensor w2, torch::Tensor b2);
"""

vit_ops = load_inline(
    name="vit_kernelbench_ops",
    cpp_sources=vit_cpp_source,
    cuda_sources=vit_cuda_source,
    functions=["patch_embed_cuda", "mlp_head_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout=0.1, emb_dropout=0.1):
        super(ModelNew, self).__init__()

        assert image_size % patch_size == 0
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
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
        self.ops = vit_ops

    def forward(self, img):
        x = self.ops.patch_embed_cuda(
            img.contiguous(),
            self.patch_to_embedding.weight.contiguous(),
            self.patch_to_embedding.bias.contiguous(),
            self.cls_token.contiguous(),
            self.pos_embedding.contiguous(),
            self.patch_size,
        )
        x = self.transformer(x)
        x = x[:, 0].contiguous()
        return self.ops.mlp_head_cuda(
            x,
            self.mlp_head[0].weight.contiguous(),
            self.mlp_head[0].bias.contiguous(),
            self.mlp_head[3].weight.contiguous(),
            self.mlp_head[3].bias.contiguous(),
        )