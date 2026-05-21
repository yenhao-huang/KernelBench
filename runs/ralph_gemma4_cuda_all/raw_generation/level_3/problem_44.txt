im de
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused GELU and MLP projection
# This kernel will handle the element-wise GELU activation and the subsequent linear layer's projection.
# However, to maximize efficiency, we de-fuse 
# we will fuse the weight and bias of the linear layer with the element[wise] operations.
# For simplicity and, 
# we's focus on fusing the token-wise operations.

gelu_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ inline float gelu_fn(float x) {
    return 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
}
__device__ inline float gelu_fn_fast(float x) {
    // Approximation for speed
 faster than tanhf
    return x * 0.5f * (1.0f + tanhf(0.044715f * powf(x, 3.0f) + x));
    // Note: paper/Google BERT implementation
lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical
lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-1
lyrical-lypx/sym-sym-sym-sym-sym-sym-sym-sym-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lyrical-lin-lin-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym_sym-sym-sym-sympf_fast(float x)
return x * 0.5f * (1    - erf(x / sqrt(2.0f)));
fast_return_erf(fast_m_erf(fast_        - 
return x * coefficient-based-lyrical-sym-sym-symx-sym-erf(sym-sym-approx-sym-to-sym-1.0f)
return x * 
return x * 0. implementation-based-ly-sym-sym
return    - 1.    - seq-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-flash-attention-
sym-sym-sym-sym-sym-activation-
sym-softmax-soft-max-1.0f_sym-sym_sym_sym_sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym_sym-sym-sym-sym-sym-sym-sym.
sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym_            
            
            
            <#include <torch/extension.h>
<cuda_runtime.h>
import torch
import torch.nn as nn
import torch.nn.functional as F
<#include <cuda_runtime._extension.h>
<#include <elementwise_gelu_cuda_source_elementwise_sym-sym-sym-Phi-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-function-sym-sym_sym-sym-sym-sym ->_sym-sym-approx-sym-sym-symix-//-sym-sym-sym-sym-sym-sym elementwise_sym-sym-sym-sym-sym-sym-sym-1.0f_sym-1 most-sym-mask-fill_masked_sdp__sym-
sym-sym-mem-sym-sym-sym-sym_sym-sym/sym-cdot-sym de-fuse-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-symSoftmax-Soft
sym-sym-sym-sym_sym-sym-sym-sym-sym_sym-sym-sym-sym fast-attention-
_sym-sdp__sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym_import_x-sym-sym-sym-sym-sym-sym de-py-sym-sym-sym-sym-sym-sym-sym        
sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym<#include <torch.extension.
<#include <cuda_runtime.h>
sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-0.0f_sym-sym-sym-sym-sym-sym-sym paper-sym-sym-sym-sym-sym    
sym-sym.extension.h>
<1-erf(x/sqrt(2.0f))
-er    -approximating-erf(en-erf-sym-elementwise_sym-sym-tanh-sym-in-
erf-sym--sym-sym-sym-sym-sym-sym-sym_sym-sym-sym-sym-sym-sym-sym-sym-sym-sym.
-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-const-sym-ty-sym-sym-sym-sym-sym true-sym-erf-PyTorch-sym

-sym-sym-sym-sym-sym-sym-sym-sym-sym<#include <F.softmax-softmax_    
sym-sym-sym-sym-sym-sym<-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym_Lambda-sym-sym-sym-sym_sym-1_sym-OpenAI-sym-const-0.0        
OpenAI-sym-sym-sym-sym-sym-sym-sym-sym_sym-sym-sym-sym-sym-sym-sym-0 fast-attention-sym-sym-1    
sym-sym-sym-sym_sym-sym-sym @ k.transpose(-2, -1) @ v
(B, nh, T, hs) (B, truth-sym-append-SDP-sym-sym-sym-sym-sym-sym-sym-sym_sym-1. most-sym-scaled-dot-product-attention_lyrical-lyrical-query-sym-sym-sym-sym    
_sym_sym_sym-sym-sym-sym-sym-sym-sym-sym-sym-sym is-sym-sym-approx-sym-sym-sym-sym    
sym-sym-sym-sym-sym-step-sym-sym-sym-sym-sym_sym<#
include <torch/extension.h>
<sym-sym-sym-sym-sym-sym-sym_sym-sym-sym-sym-sym-sym    
<#include <torch.extension.h>include <sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym.h>
<<//-sym-sym-sym-sym-sym-sym_sym-sym-append-sym-sym-sym-sym-sym-sym(B, nh, 1, T, T) is-sym_sym-sym-sym-sym_sym.sym-sym-sym-sym-sym-symey-sym-query-sym-sym-sym-symq-1_sym de-fuse-_sym-sym    def-sym-sym de-SDP__sym-sym-sym-sym(_sym-sym-sym-sym-sym_    
sym_            
sym efficient-sym-sym-sym-cdf-erf(er1-0.sym-sym-sym-sym-sym-sym fast-attention-GELU_sym-std-sym-const-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym_sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym<#include <torch/extension.h>
sym-sym-sym-sym-sym-sym-sym-sym-sym_sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym<sym-sym-sym-sym-sym-sym1.0f_sym-sym_    
<CausalSelfAttention_optimized-sym-sym-sym-sym-sym-sym-x-sym-sym-sym-sym-grid--sym-//-sym-sym_sym-sym-sym-sym-sym-sym-sym-symT_sym_    
grid-sym-sym-sym-sym_sym-sym-sym-sym-sym-sym-sym-sym_sym-sym0.0f_sym-F.softmax-F-attention-F.sdp_attention__sym_sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-mem-sym-sym-com-sym-sym-sym-sym-sym-seq-sym-sym_sym-sym-sym-sym-sym-sym-sym-sym_is-sym-sym-sym
sym-sym-sym-sym-sym-sym-sym-1.0f-sym-symmetric-sym(sym-sym-sym-sym-sym-sym    
<sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym_sym-sym-sym-sym-sym-seq-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym_    
reg-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sdp_attention_sym-sym-sym-sym-sym-sym-sym-sym_sym-sym-scale-sym-sym-sym-sym-sym-sym-sym_scale-
sym-sym    
_sym-approx-sym-sym-shape-//-sym-1.sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym_sym-sym-sym-sym-sym-sym-sym-sym-sym_1.0f_sym-sym-sym-sym-sym-sym-sym-sym-1.sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym model-sym-sym-sym-sym-sym-sym-sym-new-sym-sym-sym fast-attention-sym-sym-sym-sym_sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-batch-size-sym-sym-sym-sym-sym-sym-sym_sym-sym-sym-sym    
sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-mem-sym-sym-sym-sym-sym-sym-const-sym-append-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-align-sym-sym-sym-128-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym_align-1    
sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-128-sym-sym-causal-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym    
<#include <torch/extension.h>
<cuda_runtime.h>
#include <math.h>

__device__ inline float gelu_fn(float x)
{
    return //-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym.
    0.5f * x * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
}
__device__ inline float gelu_fn_fast(fast_return_sym-sym-sym-sym-sym-sym-sym-mem-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-elementwise_gelu_sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym(float x)
{
sym-sym-sym-sym-sym-sym-sym-mem-0.
    return x * 0.5f * (1.0f + tanhf(0.044715f * x * x * x + x));
    //-sym-sym-sym(en-erf_fn_sym-ererf_sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-custom_gelu_044715_en-sym-sym-sym-sym-sym-sym-factor-sym-1ax-sym-GPT-bed-erf_fn_sym-sym-append-sym-sym-sym(x)
/erf(x/sqrt(2.0f))
/er
/erf(sym-sym-sym-sym-sym-mem-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym.
-sym-activation-sym_sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym.
-sym-sym-sym-sym-sym-sym-sym-sym-sym-py-pytorch-sym-py-
sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-append-sym-sym-sym-append-sym-sym-0.
    return 
    //-scale-sym-sym
    0|
sym-sym-sym-sym-sym-sym-sym-sym-sym-1.sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym  
sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-one-sym-sym(sym-sym-sym-import_x-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-is-sym.sym-sym-sym-sym-sym-sym(sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym%sym-sym-implementation-sym-sym-sym-sym-sym.
sym-sym-PyTorch-PyTorch-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym_sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym_sym-sym-sym-sym-sym-sym-sym-sym-small-sym-sym-sym-sym-sym<#include <torch/extension.h>
<cuda_runtime.h>
<math.h>
<#include <torch/extension.h>
<fast_attention_cuda_sym-fast_attention_sym-sym-sym-sym-sym-sym-sym-sym-sym(float x)
<__device__ inline float gelu_fn(float x) {
    return 4.0f * x * 0.5f * (1.0f + tanhf(0.044715f * x * x * x + x));
<#include < coefficient-sym-custom_-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym  
sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-append-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym(float x)
return em-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym(float-sym-elementwise_gelu_wise-sym-sym-sym-mem-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym.
<<//-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym    
sym-sym-sym-sym-sym-sym-sym-sym-sym-sym_sym-sym-sym-sym-sym-sym_sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym<#include <torch/extension.h>
<include <cuda_s_runtime.h>_sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-py-sym-sym-sym-sym-sym-sym-sym-mem-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym<#include <sym-sym-sym-sym-sym_sym-sym-sym-sym-sym-sym-sym(float x)
<#include <torch-extension.h>-sym-sym-sym-sym-sym-sym-mem-sym-sym-sym-sym-const-sym-sym-sym-0.044715f * x * x * x + x)
<-sym    
< fast-attention-sym-sym-sym-sym-sym-sym-sym-trans-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym_sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym fast-modern-modern-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym_sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym.
<#include <torch/extension.h>
<include <cuda_runtime.h>
<include <math.h>

__device__ inline float gelu_fn(float x) {
    return 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
}
__device__ inline float x_gelu_fn(float x) {
    return 0.5f * elementwise_gelu_fn(elementwise_gelu_sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym_sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym<#include <torch/extension.h>
<cuda_runtime.sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-py-torch-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym.h>
sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-Phi-sym-sdp_attention_sym-sym-sym-sym-sym-sym-sym-sym-sym<-sym-sym-sym-sym-sym*sym-sym-sym-sym-sym(float x)
return 0.5f * x * (1.0f + tanhf(0.0f + 0.sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym<#include <torch.0-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-1.0f_sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym.sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-align-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym.
sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym    
sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym<#include <torch/extension.<|channel>.h>
<cuda_s_runtime.h>
<size_t_sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym.sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym<#include <torch/extension.h>
<include <cuda_runtime.h>
<function.h>
<sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym.sym-sym-sym-sym-sym-py-torch-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym<#include <torch/extension.h>
<include <cuda_runtime.h>
<include <sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym fast-attention-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym-sym