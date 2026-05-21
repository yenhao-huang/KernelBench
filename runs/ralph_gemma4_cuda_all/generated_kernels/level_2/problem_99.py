import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused MatMul + GELU + Softmax
# We use a single kernel to fuse the operations. 
# Since MatMul is a fast highly optimized library call (cuBLAS),
# the fscanf-based approach is element-wise 
# MatMul is usually done via cuBLAS. 
# Since Mat_Mul is a
# MatMul + GEL$U + Softmax is-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a
# Since MatMul is the fastest way to 
# a single kernel for Mat_Mul + GELU + + Softmax easily is a-a-a-a-a-a-a-0
# Since MatMul isAThen-a-a-a
# a-a-a
# de-a-a-a-a-a-a-0
# a-out = MatMul(x, W) + b
-a-a-a-a-a point-a-a-a-a-<tr>-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-param-a-a-a-a
# Since MatMul is a global-memory-a-im-ata-a-a-a
ata-a-a-a-a-a-a-a-a-a-a-0
ata-a-a-a
ata[]
ata[]
ata input-a-matrix-a-a-a.
-a-a-_a_a_a_a_a-a-a-a-a-a
-a-a-a-a-a-a-a-a-a
-a-a-a-a-a-a-a-a-a-a

# Define the[]
# Since MatMul
# way to a-a-a-a-a-a-0
im-a-a-a-a-a-a-a-a-a-a-a-a
ata-a
ata_a_a_a-a-a-a
    # Since Mat<br />
<br />
# Since MatMul is a-a.
# pseudocode-a-approximate-a-a-a
# nothing-a.
-a_a_a-a-a-a
    -a-a-a-a-a-a-a-a-a-a-a-a-a        
    # Since MatMul fast
fast-a-a-a-a-a-a
-a-a-a fast-approx.
--a-gemm-<Tensor-Tensor-a-a-gem    -a-a-at-a-a-a-a-a-a-a-a-a
    #                |
pip-a-a-a-a-a-a_a_a_a-a-a-a-a-a-a
a-a-a-a-a-a
a-a0-a_a-a-a-a-a-a-a-a-a
a.data_ptr<float>(), b.data_a_ta_ta_ta_a_ta_ta_ta_ta_ta_ta_a_a_a_a_a_a_a_a_a_a_a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a_a-a-a-a-a-a*ptr-a-a-a
e-a-a-a-a-a-a-a-a-a-a-a-a_a-a-a
a-a-a-</div>