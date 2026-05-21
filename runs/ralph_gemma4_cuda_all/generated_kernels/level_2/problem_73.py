import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Conv2d + BatchNorm2d + Scaling
# Since implementing a full Conv2d from scratch in a
# highly optimized cuDNN-based Conv2d is cuDNN call,
# implementing a
# fusion of Conv2d + BN + Scale is Conv2d + (BN_fused_params)
#
# We can optimize by fusing the                
# BN_f. = (x - mean) / sqrt(var + eps) * gamma / scale_factor
#
# We. can transform the BN_import (x * gamma_fused) + beta_f

- (x - mean) / sqrt(var + eps) * gamma / scale_factor
- (x_conv) * (gamma / (sqrt(var + eps) * scale_factor)) + (beta - mean * gamma / (sqrt.var + eps) * scale_factor)
- Let W_fused- (W__conv)
- Let W_ element-wise element-wise element-wise element    
- Let W_fused = W_conv * (gamma / (sqrt(var + eps) * eps_scale) scale_factor
- Let W
-               - mean * gamma / (sqrt(var + eps) scale_eval) scale_factor

- Let B_fused = B_    - mean * (gamma/ (sqrt_var_eps) * scale_float)
-
 weight_grad_training- weight_ folding- folding- folding-
 weight_ inference- inference-scale_factor
    
# We will implement a
 ways to optimize:
- 
-
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Conv2d + BatchNorm2d
# In inference-mode, terms: output = (input * weight_fused) + bias__fused
- (x - mean)
- / sqrt(var + eps) std_dim = (gamma / sqrt(var
- factor = gamma / (sqrt(std_dev_        + eps) * scaling_factor)
_fused_factor = (gamma / (sqrt(var + eps))) * (1 / scaling_factor)
        - (beta - mean * (gamma / sqrt(var + eps))) * (1        / scaling.factor)
        _fused_scale = gamma / (sqrt(var + eps) * scaling_factor)
    _fused_bias = (beta - mean * gamma / sqrt(var + eps)) / scaling_factor
    # Wait, letthought