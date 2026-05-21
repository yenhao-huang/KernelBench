import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused operations: matmul + scale + residual + clamp + logsumexp + mish
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_kernel_forward(
    const float* input, const float* weight, const float* bias,
    float* output, float* logsumexp_out,
    int batch_size, int input_size, int hidden_size,
    float scale_factor, float clamp_min, float clamp_max) {

    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;
    float* shared_weight = shared_mem + input_size;

    int row = blockIdx.x;
    if (row >= batch_size) return;

    // Load weight row into shared memory (coalesced)
    int tid = threadIdx.x;
    for (int i = tid; i < input_size; i += blockDim.x) {
        shared_weight[i] = weight[row * input_size + i];
    }
    __syncthreads();

    // Compute matmul for this row
    float sum = 0.0f;
    for (int i = 0; i < input_size; i++) {
        sum += input[row * input_size + i] * shared_weight[i];
    }
    sum += bias[row];  // Add bias

    // Scale
    sum *= scale_factor;

    // Residual connection (x = x + x, which is 2*x)
    sum = sum + sum;

    // Clamp
    sum = fminf(fmaxf(sum, clamp_min), clamp_max);

    // Store intermediate result
    output[row] = sum;

    // Compute logsumexp (single element, so just the value itself)
    // Since we're reducing over dim=1 with keepdim=True, and hidden_size=1 after matmul,
    // logsumexp of a single element is just that element
    float logsumexp_val = sum;
    logsumexp_out[row] = logsumexp_val;

    // Mish activation: x * tanh(softplus(x))
    // softplus(x) = log(1 + exp(x))
    // For numerical stability, use: log1p(exp(x)) when x < 20, else x
    float sp;
    if (sum < 20.0f) {
        sp = log1pf(expf(sum));
    } else {
        sp = sum;
    }
    float tanh_sp = tanhf(sp);
    float mish_result = sum * tanh_sp;

    // Multiply logsumexp with mish result
    output[row] = logsumexp_val * mish_result;
}

torch::Tensor fused_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float scale_factor, float clamp_min, float clamp_max) {

    int batch_size = input.size(0);
    int input_size = input.size(1);
    int hidden_size = weight.size(0);

    auto output = torch::empty({batch_size, hidden_size}, input.options());
    auto logsumexp_out = torch::empty({batch_size, 1}, input.options());

    const int threads = 256;
    const int blocks = batch_size;
    const int shared_mem_size = input_size * sizeof(float) * 2;  // For input and weight

    fused_kernel_forward<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), logsumexp_out.data_ptr<float>(),
        batch_size, input_size, hidden_size,
        scale_factor, clamp_min, clamp_max);

    return output;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float scale_factor, float clamp_min, float clamp_max);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.fused_op = fused_op

    def forward(self, x):
        # Use custom fused CUDA kernel that performs all operations
        return self.fused_op.fused_cuda(
            x, self.matmul.weight, self.matmul.bias,
            self.scale_factor, self.clamp_min, self.clamp_max
        )