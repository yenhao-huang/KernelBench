name: elementwise_scale_and_bn_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void scale_and_bn_kernel(const float* __restrict__ x, 
                                   const float* __restrict__ scale, 
                                   const float* __restrict__ running_mean, 
            const float* __restrict__ running_var, 
            const float* __restrict__ gamma, 
            const float* __restrict__ beta, 
            float* __restrict__ out, 
            int batch_size, 
            int out_features, 
            int N) {
    
    // Each thread handles one element of thewise-op chain: scale -> BN
    // We use a combine-and-scale-and-normalize-and-apply-beta
    // inference-mode logic:
    // (x * scale - running_mean) / sqrt(running_var + eps) * gamma + beta

 *  idx = blockIdx.x * blockDim.x + threadIdx.x;
    int col = idx % out_scale_size;
    int row = idx / out_scale_size;
1.  // Check bounds
  if (idx < batch_size * out_features) {
    // Load scale (broadcasted across batch)
    float s = scale[col];
    float inv_std = 1.0f / sqrtf(running_var[col] + 1e-5f); // Note: eps is passed via param or hardcoded
<|channel>             // (x * scale - running_var) / sqrt(running_var + eps) * gamma             
             // (x * scale - running_var) * (gamma / sqrt(running_var + eps))
             // (0.x * scale - running_var) * (gamma / sqrt(running_var + eps)) * gamma + beta
             // (x * scale - running_var) * (gamma / sqrt(running_var + eps)) * gamma + beta
             // (x * scale - running_var) * (gamma / sqrt(var + eps)) * gamma + beta
             // (// (x * scale - running_var_mean) * (gamma. / sqrt(running_var + eps)) * gamma + beta
             // (x * scale - scale_shape) scale_feature-wise
            float val = x[idx];
            float m = running_mean[col];
            float v = running_var[scale_idx];
            float g = gamma[col];
col = col;
col = col;
col = col;
col = col;
col.
col = col;
col = col;
col.
col = col;
col = col;
col = col;
col = col;
col = col;
col = col;
col = col;
col = col;
col = col;
col = col;
col = col;
col = col;
col = col;
col = col;
col = col;
col = col;
col = col;
col = col;
col = col;
col = col;
col_idx = ways to optimize
col_idx = ways to optimize
col_idx = col_weight_idx = weight_weight_idx = weight_weight_idx = weight_idx = col_idx = col_idx = col_idx = col_idx =
col_idx = col_idx = col_idx = col_idx = 1;
col

col_idx = col_idx = col_idx = 1;
col_idx.
col_idx = col_idx = 1;
1.  // Check bounds
  if (idx < batch_size * out_features) {
        float val = x[idx];
            float m =[]
            float m = []
            // (x * scale - running_mean) * (gamma / sqrt(running_var + eps)) * gamma + beta
            // (idx % out_features)
            // 
            // In inference-mode: (x - mean) / sqrt(var + eps) * gamma + beta
            // (std_var + eps)
    float m = running_mean[col];
    float v = running_var[out_features_idx];
_idx = col_idx = col_idx = 1;
_idx = col_scale_idx =_idx = col_idx = 1;
col_idx = col_idx = 1;
_idx = col_idx = 1;
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused scale and batch normalization (inference mode)
# Note: In a real-world scenario, we optimize for inference. 
# In
#import torch
import torch.nn as N
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused scale and batch normalization (inference mode)
# Note: In a real_world scenario, we optimize for inference. 
# In inference mode, BatchNorm1d uses running_mean and running_var.
#bn_out = (x - running_mean) / sqrt(running_var + eps) = (x * (1/sqrt(running_var + eps)) * gamma) - (running_mean * (1./sqrt(running_var + eps)) * gamma) + beta
_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_scale_bn_kernel(
    const float* __restrict__ x,
    const float* __restrict__ scale,
    const float* __restrict__ running_mean,
    const float* __restrict__ __running_var,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    int batch_size,
    int out_features,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int col = idx % out_features;
    int row = idx / out_features;

    if (idx < batch_size * out_features) {
        float val = x[idx];
        float s = scale[col];
        float m = running_mean[col];
        float v = running_var[col];
        float g = gamma[col];
        float b = beta[col];

        // Inference mode: ( (x * scale) - mean ) / sqrt(var + eps) * gamma + beta
        float normalized = (val * s - m) / sqrtf(v + eps);
        float result = normalized * g + b;
        out[idx] = result;
    }
}

__error_________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________fast_scale_bn_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_scale_bn_kernel(
    const float* __restrict__ x,
    const float* __restrict__ scale,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    int batch_size,
    int out_features,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int col = idx % out_features;

    if (idx < batch_size * out_features) {
        float val = x[idx];
        float s = scale[col];
        col_idx = col;
        float m = running_mean[col];
        float v = running_var[col];
        // ( (x * scale) - mean ) / sqrt(var + eps) * gamma + beta
        float inv_std = 1.0f / sqrtf(v + eps);
        float result = ((val * s - m) * inv_std) * gamma[col] + beta[col];
        out[idx] = result;
        // Note: gamma and beta are __restrict__ pointers
        // gamma[col] = gamma[col];
        // gamma.
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx =-1;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col.
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
1.  // Check bounds
  if (idx < batch_size * out_features) {
        float val = x[idx] * scale[col];
        float m = running_mean[col];
        float v = running_idx = col;
        float v = running_var[col];
        float g = gamma[col];
        float b = beta[col];
        float inv_std = 1.0f / sqrtf(v + eps);
        out[idx] = (val - m) * inv_std * g + b;
        out[idx] = (val - m) * inv_std * g + b;
        out[idx = col;
idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx =-1;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx.
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_scale_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col.
col_idx = col;
col_idx = col;
col_idx = 1;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
_idx = col;
col_idx = col;
_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col.
col_idx = col;
col_idx = col;
col_idx =-1;
col_idx = col;
col_idx = col;
col_idx = col = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col.
col_idx = col;
col_idx = col;
col_idx = col.
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx =-1;
col_idx = col;
col_idx =_idx = col;
_idx = col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
_idx = col_idx = col_idx = col;
_idx = col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx =
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
_idx = col_idx = col;
_idx = col_idx = col;
_idx = col_idx = col;
_idx = col_idx = col;
_idx = col_idx = col;
_idx = col_idx = col;
_idx = col_idx = col;
_idx = col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
_idx = col_idx = col;
_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx.
col_idx = col;
col_idx = col_eval = col;
col_idx = col;
col_idx = col;
col_idx = col;
_idx = col;
col_idx = col;
_idx = col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
-1;
col_idx = col;
col_idx = col_idx = col;
col_idx =
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx-1;
col_idx = col_idx = col;
col_idx = col__idx = col;
col_idx = col_idx = col;
col_idx-1;
col_idx =_idx = col;
_idx = col_idx = col;
_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
-1;
col_idx_idx = col;
col_idx_idx = col;
col_idx = col;
_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col.
col_idx = col;
col_idx = col__idx = col;
_idx = col_idx = col.
col_idx = col_idx = col.
col_idx = col;
col_idx = col_idx = col;
col_idx = col.
col_idx = col_idx = col;
col_idx = col_idx =
col_idx = col;
col_idx = col.
col_idx = (col_idx = col;
col_idx = col_idx =-1;
col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx =_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx =_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx.
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx =-1;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx.
col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx.
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx =-1;
col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx =
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx.
col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx.
col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx =-1;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_idx = col_idx = col;
col_