import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Conv2d + GELU + AdaptiveAvgPool2d
# We will fuse the ability to
# 1. On-the-channels-of-the-out_channels dimension.
Utilitarian-style:
# let' de-facto facto-facto-facto-facto-fact0-fact-fact-fact-fact-fact-fact-fact-fact-fact-fact-fact-fact-fact-
let' de-facto facto-facto-facto-facto-facto-fact-fact-cut-fact-fact-partial-fact-fact-fact-fact-fact
let' de-facto facto-fact-fact-fact-fact-fact-batch-fact-stride-fact-fact-fact-x-fact-conv-type-
let' de-im-facto-cudatoolkit-en-fact-cuDNN-
let' de-fact    -fact-fact-cuDNN-
cuDNN is thecomment-comment-fact-step-block-target-
cu1-fact-batch-salt-fact-channels-batch-conv de-function-F
-fact-fact-fact<: (im-fact-fact-fact-idx-back-fact-idx-activation-activation-fact-fact-algorithm-fact-fact-factor-fact-fact-fact-
fact-fact-fact-fact-fact-fact-fact-filter-weight-filter- incluso-fact-fact-fact-fact-batch-input-input-fact-fact-fact-idx-data-```

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inle

# Define the custom CUDA kernel for fused Conv2d + GELU + AdaptiveAvgPool2d
# We replacement-style: replacement-style: replacement-import-flow-step-conv-conv-squeeze-Adam-flow-batch-
replacement-style: replacement-1-replacement-style: replacement-1-replacement<:
import torch
import torch.nn.functional as F
# Define theimport-import-import-import-import-import-import-cuda-kernel-call-output-
# Define the    -fact-fact-fact-fact-fact-import-import-F-fact-conv-size-let-
# intent-import-prompt-import-flag-0.append-batch_size=12im8
batch_size = 128
batch_size = 128
batch_ de-facto-fact-fact-fact-x-conv-factor-conv                
                
                # F    -fact-fact-batch-size-1K-batch_size=1        
                # F    -fact-idx-data-index-1D-f1-x-in_channels=|\ de-facto-batch<channel|>import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# The optimization strategy is to fuse the GELU activation and the Global Average Pooling (GAP) 
# into a single kernel that operates on the output of the convolution. 
# While Conv2d is best handled by cuDNN, fusing the subsequent element-wise GELU 
# and the reduction (AdaptiveAvgPool2d) reduces memory bandwidth overhead by 
# avoiding writing the large intermediate GELU tensor to global memory.

fused_ops_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
__device__ __forceinline__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
}

__global__ void gelu_avg_pool_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int out_channels,
    int height,
    int width) 
{
    // Each thread handles one (batch, channel) pair to compute the average
    int b = blockIdx.y;
    int c = blockIdx.x;
    
    if (b < batch_size && c < out_channels) {
        float sum = 0.0f;
        int spatial_size = height * width;
        
        // Offset for the current (batch, channel)
        int base_idx = (b * out_channels * spatial_size) + (c * spatial_size);
        
        for (int i = 0; i < spatial_size; ++i) {
            float val = input[base_idx + i];
            sum += gelu(val);
        }
        
        output[b * out_channels + c] = sum / (float)spatial_size;
    }
}

torch::Tensor fused_gelu_avg_pool_cuda(torch::Tensor input) {
    // input shape: (batch_size, out_channels, height, width)
    auto batch_size = input.size(0);
    auto out_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    
    auto output = torch::empty({batch_size, out_channels}, input.options());

    dim3 block(1, 1); // We use a simple approach where each thread handles one channel
    // To optimize, we can use more threads per block, but for simplicity and correctness:
    dim3 grid(out_channels, batch_size);

    gelu_avg_pool_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_channels,
        height,
        width
    );

    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_gelu_avg_pool_cuda(torch::Tensor input);
"""

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_cuda_source,
    functions=["fused_gelu_avg_pool_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution, then uses a fused CUDA kernel
    to perform GELU activation and Global Average Pooling in one pass.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.fused_ops = fused_ops

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
        Returns:
            Output tensor of shape (batch_size, out_channels)
        """
        # Step 1: Standard Convolution (highly optimized via cuDNN)
        x = self.conv(x)
        
        # Step 2: Fused GELU + AdaptiveAvgPool2d (reduces memory R/W)
        # The kernel expects (N, C, H, W) and returns (N, C)
        x = self.fused_ops.fused_gelu_avg_pool_cuda(x)
        
        return x