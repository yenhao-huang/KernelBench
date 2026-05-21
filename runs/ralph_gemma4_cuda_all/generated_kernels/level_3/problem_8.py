import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused Conv+BN+ReLU
# This kernel will handle the addition of identity and the second ReLU
# This kernel will also handle the addition of identity and the second ReLU
# This kernel and will be fused-up.
# We will fuse the Conv2d + BatchNorm2d + ReLU
# We de-fuse and de-fuse.
# We de-fuse.
# We}")

# Define the custom CUDA kernel for fused Conv+BN+ReLU
# This kernel will handle the-
# This kernel will handle the identity and thecu-
# This kernel will be fused-up.
# This kernel de-fuse.
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import lambda_extension_extension_extension_extension_extension_extension_extension_extension_extension_extension_extension_extension_idx_idx_extension_extension_exception_exception_exception_conv_conv_conv_conv_conv_conv_idx_id_idx_idx_idx_im_idx_extension_text_extension_idx_idx_id_cudacode______________F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_Res_F_F_F_F_F_F_F_F feeling_feeling_F_F_F_F_out_F_out_conv_PyTorch_import_F_import_F_import_F_import_Res_import_ReLU_import_F_std_import_import_F_data_import_conv2d_F_out_Res_name="ModelNew"
name = "Model        
```python
import torch
import torch.nn as nn
importimport_F_F_import_F_F_import_F_F_import_F_F_import_F_F_import_F_F_import_F_F_import_F_F_import_F_F_import_F_F_import_F_F_import_F_F_import_F_F_import_F_F_import_F_F_import_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F
import_F_F_import_F_F_import_F_F_F_F_F_F_F_F_F_F_F_F_F_batch_size_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_batch_F_F_F fast_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_F_FF_F_F_F_F_F_F * F_F_F_F_F_im_F_<|channel>thought
<channel|>```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused residual addition and ReLU
# This kernel performs: out = relu(out + identity)
# This reduces memory bandwidth by performing the addition and activation in one pass.
fused_residual_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_residual_relu_kernel(float* out, const float* identity, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = out[idx] + identity[idx];
        out[idx] = val > 0.0f ? val : 0.0f;
    }
}

void fused_residual_relu_cuda(torch::Tensor out, torch::Tensor identity) {
    int size = out.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_residual_relu_kernel<<<num_blocks, block_size>>>(
        out.data_ptr<float>(), 
        identity.data_ptr<float>(), 
        size
    );
}
"""

fused_residual_cpp_source = """
void fused_residual_relu_cuda(torch::Tensor out, torch::Tensor identity);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_residual_cpp_source,
    cuda_sources=fused_residual_source,
    functions=["fused_residual_relu_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )
        self.stride = stride
        self.fused_ops = fused_ops

    def forward(self, x):
        identity = x

        # First block: Conv -> BN -> ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second block: Conv -> BN
        out = self.conv2(out)
        out = self.bn2(out)

        # Shortcut connection
        if self.downsample is not None:
            identity = self.downsample(x)

        # Fused Residual Addition and ReLU: out = relu(out + identity)
        # We use the custom CUDA kernel to perform this in a single kernel launch
        # to minimize memory round-trips.
        self.fused_ops.fused_residual_relu_cuda(out, identity)

        return out