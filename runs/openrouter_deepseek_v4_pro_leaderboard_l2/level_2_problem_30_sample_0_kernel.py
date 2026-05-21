import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused GEMM + GroupNorm + HardTanh
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_gemm_groupnorm_hardtanh_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    int num_groups,
    float hardtanh_min,
    float hardtanh_max
) {
    // Each block handles one output element (row, col)
    int row = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < out_features) {
        // Compute GEMM for this element
        float sum = bias[col];
        const float* input_row = input + row * in_features;
        const float* weight_col = weight + col * in_features;
        
        for (int k = 0; k < in_features; ++k) {
            sum += input_row[k] * weight_col[k];
        }
        
        // Group Normalization
        int group_size = out_features / num_groups;
        int group_id = col / group_size;
        int group_start = group_id * group_size;
        int group_end = group_start + group_size;
        
        // Compute mean and variance for this group in this sample
        float mean = 0.0f;
        float var = 0.0f;
        
        // First pass: compute mean
        for (int c = group_start; c < group_end; ++c) {
            float val = 0.0f;
            if (c == col) {
                val = sum;
            } else {
                // Recompute GEMM for other columns in the group
                float temp_sum = bias[c];
                const float* w_col = weight + c * in_features;
                for (int k = 0; k < in_features; ++k) {
                    temp_sum += input_row[k] * w_col[k];
                }
                val = temp_sum;
            }
            mean += val;
        }
        mean /= group_size;
        
        // Second pass: compute variance
        for (int c = group_start; c < group_end; ++c) {
            float val = 0.0f;
            if (c == col) {
                val = sum;
            } else {
                float temp_sum = bias[c];
                const float* w_col = weight + c * in_features;
                for (int k = 0; k < in_features; ++k) {
                    temp_sum += input_row[k] * w_col[k];
                }
                val = temp_sum;
            }
            float diff = val - mean;
            var += diff * diff;
        }
        var /= group_size;
        
        // Normalize
        float eps = 1e-5f;
        float inv_std = rsqrtf(var + eps);
        float normalized = (sum - mean) * inv_std;
        
        // Scale and shift
        float gn_out = gamma[col] * normalized + beta[col];
        
        // HardTanh
        float result = gn_out;
        if (result < hardtanh_min) result = hardtanh_min;
        if (result > hardtanh_max) result = hardtanh_max;
        
        output[row * out_features + col] = result;
    }
}

torch::Tensor fused_gemm_groupnorm_hardtanh_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gamma,
    torch::Tensor beta,
    int num_groups,
    float hardtanh_min,
    float hardtanh_max
) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::empty({batch_size, out_features}, input.options());
    
    const int threads_per_block = 256;
    dim3 blocks(batch_size, (out_features + threads_per_block - 1) / threads_per_block);
    
    fused_gemm_groupnorm_hardtanh_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        num_groups,
        hardtanh_min,
        hardtanh_max
    );
    
    return output;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_gemm_groupnorm_hardtanh_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gamma,
    torch::Tensor beta,
    int num_groups,
    float hardtanh_min,
    float hardtanh_max
);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_gemm_groupnorm_hardtanh",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_gemm_groupnorm_hardtanh_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max)
        self.fused_op = fused_op
        self.num_groups = num_groups
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

    def forward(self, x):
        return self.fused_op.fused_gemm_groupnorm_hardtanh_cuda(
            x,
            self.gemm.weight,
            self.gemm.bias,
            self.group_norm.weight,
            self.group_norm.bias,
            self.num_groups,
            self.hardtanh_min,
            self.hardtanh_max
        )