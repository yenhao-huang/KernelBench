import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Hinge Loss
# The operation is: mean(clamp(1 - predictions * targets, min=0))
# This can be fused into a single kernel to avoid multiple memory passes.
hinge_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hinge_loss_kernel(const float* predictions, const float* targets, float* out_sum, int size) {
    // Use atomicAdd to accumulate the sum of the hinge loss values
    // Each thread processes one element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = 1.0f - predictions[idx] * targets[idx];
        if (val < 0.0f) {
            val = 0.0f;
        }
        atomicAdd(out_sum, val);
    }
}

torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto size = predictions.numel();
    
    // We need to compute the sum of the hinge losses first
    // Allocate a single float on device for the sum
    float h_sum = 0.0f;
    float* d_sum;
    cudaMalloc(&d_sum, sizeof(float));
    cudaMemset(d_sum, 0, sizeof(float));

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    hinge_loss_kernel<<<num_blocks, block_size>>>(predictions.data_ptr<float>(), targets.data_ptr<float>(), d_sum, size);

    // Copy result back to host
    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_sum);

    // Return the mean
    return torch::tensor(h_sum / size, predictions.options());
}
"""

hinge_loss_cpp_source = (
    "torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets);"
)

# Compile the inline CUDA code for Hinge Loss
hinge_loss_module = load_inline(
    name="hinge_loss",
    cpp_sources=hinge_loss_cpp_source,
    cuda_sources=hinge_loss_source,
    functions=["hinge_loss_cuda"],
    verbose=False,
    extra_cflags=["-O2"],
    extra_ldflags=["-lcudart"],
)


class ModelNew(nn.Module):
    """
    A model that computes Hinge Loss for binary classification tasks using a custom CUDA operator.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.hinge_loss_op = hinge_loss_module

    def forward(self, predictions, targets):
        return self.hinge_loss_op.hinge_loss_cuda(predictions, targets)