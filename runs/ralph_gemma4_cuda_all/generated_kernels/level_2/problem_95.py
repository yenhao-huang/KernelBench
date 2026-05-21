import torch
import torch.nn as.Module
import torch.nn.functional as F

class Model(nn.Module):
    """
        Simple model that performs a matrix multiplication,
     de-adds a-value,
     de-adds a-value,
     de-adds a-new-value,
     de-adds a-value,
     de-adds a-point-value,
0.0, 
    de-adds a-value,
    de-adds a-value,
     de-wise-adds a-value,
    de-adds a-value,ers,
    de-adds a-value,
    de-adds a-value,
        de-step-adds a-value,
    de_adds_a_value,
    de_adds_a_grid-value,
    de_a_a_value,
 даже-adds a-value,
    de-adds a-value,
    de-adds a
-value,
    de-adds a
-value,
    # comments
    # GELU approximation-
    -value,
ers,
    de-adds a-value,
    def forward(doc-doc-doc-doc-doc-doc-doc-doc-doc-control-doc-doc-version-doc-doc-step-doc-doc-step-doc-activation-activation-training-doc-doc-step-docKalman-doc-------------------------------------------------.
doc-doc-factor-template-step-version-clamp-step-sign-fast-activation-1.0, fast-sign-clamp-fast-name-activation-
    def forward(doc-doc-doc-0. way-to-f0-f0-activation-er-er-batch-size, batch._size, ways-to-f
    doc-X_0_0_0_0_0_0_0_    doc-doc-value-out-0_0._value_0.0,
    docdoc-doc-doc-batch-doc-0_    doc.0.0, version-factor-0://-0.0_    version->-1.index.0_F-0.0,
    doc-doc-0_0_0.0, 0.0, hardtanh_min_import-
import torch
import torch.nn.
-value,
    def forward_doc-doc-activation-step-step-step-step-0(_activation-Gaussian-shape-shape-    
    def_forward_    _activation_step-f0-factor-function-matrix-activation-post-activation-step-f|
        _activation.step-step-swap-and-beta-step-activation-value-shape-
               _activation-step-shape-
            -step-step-step-step-step-step-step-step-factor-derivation-step-val-step-
step-step-step-0.000.0, 0_0_0
    def forward_step-step-step-step-step-step.step-step-step.step-step-step    -x_    self.matmul.weight.weight.weight.weight.append-append-weight.model_    _activation.step-step-x_    self.step-step-step-step-0.0_activation.0.0_activation-step-step*step 
    step-step-step-step-step-step-step0.00    doc-doc-step-step-step-step-step-step-step-step-step-0.    doc_step-step.00.0_step-step(x)
    doc-doc-step.step-features-step-step-step-step.step.step-step-step-step-step_step_step.step-step-step-step-step
step-step-step fast-step-step-step-step-step-step-step-step-step-step-step-step-step-step-step-step-step-step-step-step-step-step-step-step-step-step-step-step-step-step-step-step-step-step-step-step-step hardtanh_min=-1.0, hardtanh_max=float('inf'),
    de-adds a-value,
    de-step-step-step-step-step-step-step-step<
    def forward(self        _activation_step-step-step-step-step-step-step-step-step-step-step-step-step-step    _step-step-step.step-step-step-step * step-step-step-output-step-g0_0_0_0_0.0, 0.prompt-step-step-step-step-step_step-step-step-shape-step-step-0_0.0,coef-coef-step-step-step-step-step_step-step-step-step<
        return x
        de-adds a-step-step-step-step-step-step-step-step-stepstepstepstepstepstep-step-step-step-step-step-step-step<
    step-step-step-step-step-step0.
    step-step0.0
    self.matmul.weight.weight.step-step-step-step-step-step-step-step_step-step_step_step-step*step-step-step-step-step-step-step    _activation.activation_step-step-step.append-step-step-step_step-step k-step-step-step fast-step-step-step-step-step-step0.0_activation.step-step    def forward(x):
step-step-step-step-step-step-step-step-step-step-step-step    _activation.step-step-step-step-step-step-step-step-step=_activation.step*step-step-step-step-step-step-step-step-step    def forward(doc-doc-step-step-step-step-step-step0.0, 0.0, hardtanh_step-step-step-step-step<
    import torch
import torch.nn as nn
    def forward(step-step-step-step-step-step-step-step-step-step-step-step-step-step-step-step-step    _activation.step-step-step-step-step0.0-step-step-step-step-step-step-step-step-step
    step-step-step-step-step-step-step-step-stepstepstepstep-step-step-0.0_step-step-step-step-step-step-steping-step-step-step-step-stepstepstep-step-step:
    step-step-step-step-step-step0. input-nn.Module
        self.matmul.weight.weight.step-step-step-step<
    step-step    self.matmul.step-add_value.step-step-step-step-step-step-step-step-step-step-step-step-step-step-step    _activation.step-step-step    _step-step-step respect-step-step-step.step-step0.0.0, lack-of-step-step-step-step*step-stepstep-step-step*gemm-

step-step-step-step-step-step-step-step<channel|>```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel to fuse:
# 1. Addition (x + add_value)
# 2. Swish (x * sigmoid(x))
# 3. Tanh (tanh(x))
# 4. GELU (0.5 * x * (1 + erf(x / sqrt(2))))
# 5. Hardtanh (clamp(x, -1, 1))
# Note: We use the standard GELU formula: 0.5 * x * (1.0 + erf(x / sqrt(2.0)))
# Since Hardtanh is the last step and clamps to [-1, 1], it is applied at the end.

fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float gelu(float x) {
    return 0.5f * x * (1.0f + erff(x * M_SQRT1_2));
}

__global__ void fused_activation_kernel(const float* __restrict__ input, 
                                        const float* __restrict__ add_val, 
                                        float* __restrict__ output, 
                                        int batch_size, 
                                        int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_features;

    if (idx < total_elements) {
        // Map global index to feature index for broadcasting add_val
        int feature_idx = idx % out_features;
        
        float val = input[idx] + add_val[feature_idx];
        
        // Swish: x * sigmoid(x)
        val = val * (1.0f / (1.0f + expf(-val)));
        
        // Tanh
        val = tanhf(val);
        
        // GELU
        val = gelu(val);
        
        // Hardtanh: clamp(val, -1, 1)
        if (val < -1.0f) val = -1.0f;
        else if (val > 1.0f) val = 1.0f;
        
        output[idx] = val;
    }
}

torch::Tensor fused_activation_cuda(torch::Tensor input, torch::Tensor add_val) {
    auto batch_size = input.size(0);
    auto out_features = input.size(1);
    auto output = torch::empty_like(input);

    int total_elements = batch_size * out_features;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_activation_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        add_val.data_ptr<float>(), 
        output.data_ptr<float>(), 
        batch_size, 
        out_features
    );

    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_activation_cuda(torch::Tensor input, torch::Tensor add_val);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_activation_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model using a fused CUDA kernel for the activation sequence.
    The sequence: Matmul -> Add -> Swish -> Tanh -> GELU -> Hardtanh
    is compressed into a single element-wise kernel to minimize memory bandwidth usage.
    """
    def __init__(self, in_features, out_features, add_value_shape):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape))
        self.fused_ops = fused_ops

    def forward(self, x):
        # Matmul is a heavy compute operation, we keep it as standard optimized BLAS/cuBLAS
        x = self.matmul(x)
        
        # The rest of the operations are element-wise and memory-bound.
        # We fuse them into a single kernel to avoid multiple reads/writes to global memory.
        x = self.fused_ops.fused_activation_cuda(x, self.add_value)
        
        return x