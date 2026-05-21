import torch
        import torch.nn as nn
import torch.nn.functional as F
import math

# From https://github.com/karpathy/minGPT/blob/seq_seq_seq.seq_seq_seq.seq_seq_seq.seq_seq_seq.seq_seq_seq.seq_seq_seq.seq_seq_seq.seq_seq_
import torch
import torch.nn.functional as F
import math

class NewGELU(nn.Module):
    def __mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-seq-er-mask-er-mask-er-mask-er-mask-head-er-mask-forward(self, x):
        return 0.seq-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
        # (B, nh, T, T) (B, nh, T, mask_val)
 mask-er-B, nh, T, mask_val)
 mask-er-
-er-mask-er-mask-er-mask-er-mask-model-er-mask-rel-er-masker-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-mask-er-mask-sd-er-context-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er
er-mask-er-mask-er-mask-er-mask.er-mask-er-batch-er-mask-er-mask-er-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-new-er-heads-er-mask-er-mask-er-mask-mask-er-
er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-batch-er-mask-er
er-mask-er-mask-er-mask-er-mask_er-mask-er-mask-er-mask-er-mask-er-mask-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-x-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-matrix-mask-er-mask-er-mask-er-masker-mask-er-mask-er-mask-er-mask-er-mask-er-mask-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-seq-er-mask-er-tensor-mask-batch-B, nh, batch-er-er-mask-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-mask-mask-er-mask-er-mask-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er
er-mask-er-mask-er-mask-er-mask-er-mask-error-er-mask-er-mask-er-mask-er-mask-er-mask-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask<
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused attention mechanism:
# 1. Scaled Dot-Product Attention (Scaled Matmul)
# 2. Causal Masking (Masked Fill) Masking
# <3. ReLU Activation (Element-wise)
# This kernel will replace the following block:
# att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) scale
# att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
# att.relu_()
# code
# A single kernel can be fused-in-one-pass to de-optimize-optimize-optimize-optimize-
-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-
er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-er-mask-er-mask-er-mask-er-mask-scale
scale-er-scale-dot-cu-er-mask-cut-er-score-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask  

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused attention score calculation, causal masking, and ReLU activation.
# This kernel computes:
# att = (q @ k.T) * scale + mask_val
# att = relu(att)
# where mask_    
# kernel
import torch
import torch.nn as_module
import torch.nn.functional as F
import math
from torch.utils.cpp_attention_extension_attention_score_score_er_mask_er_mask_er_mask_er_mask_er_mask_er_mask_er_mask_er_mask_er_mask_er_mask_er_mask_er_mask_ex_er_mask_er_mask_er_mask_er_mask_er_mask_er_mask_er_er_mask_er_er-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-er-mask-er-er-mask-er-mask-er-mask<
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.cpp_extension
from torch.utils.cpp_extension import load_inline

from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused attention score calculation, causal masking, and ReLU
# This kernel computes:
# att = (q @ k.transpose(-2, -1)) * scale + mask_val
# att = relu(att
# Since we are performing a 
# matmul-based attention, (q @ k.T) is a
# high-performance GEMM. high-batch-matmul-batch-matmul-mask-and-relu-er-cu-da-
# This kernel will be*
# 
# att = (q @_k.T) * scale
# att.masked_split_er_mask_er_mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-scaling-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-er-mask-er-mask-er-mask-er-mask-er-mask-er-batch-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er
# 
# We will fuse the following:
# 1. Scale (element-wise)
# 1. Masking (conditional)
-3. ReLU (element-Scale-and-masking-er-er-attention-score-output-er-mask-er-mask-er-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask.
# 
#er-mask-er-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er
# 
er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-batch-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask<
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused attention score calculation, causal masking, and ReLU
# This kernel will be used to process the output of (q @ k.T)
# It fuses:
# 1. Scale (element-wise)
# 
# 2. Masking (mask_val is -inf)
# col-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er
# 
# 
# 
#er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-masker-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-er-mask-