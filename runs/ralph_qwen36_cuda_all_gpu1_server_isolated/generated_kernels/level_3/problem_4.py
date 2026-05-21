import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernels for LeNet-5 optimization
# We will fuse Conv2d + ReLU, and optimize the Linear layers with fused Matmul+ReLU

custom_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s at line %d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// 1. Conv2d + ReLU Kernel
// Assumes NHWC layout for simplicity in manual implementation or NCHW with specific indexing.
// Here we implement a standard im2col + gemm approach fused with ReLU is complex.
// Instead, we use a direct convolution kernel which is often faster for small kernels like 5x5 on modern GPUs 
// if implemented well, but for simplicity and correctness in inline code, 
// we will rely on the fact that PyTorch's conv2d is highly optimized.
// However, to demonstrate "custom operator", let's implement a fused Conv2d+ReLU using a simplified direct convolution logic 
// or simply replace the FC layers which are often bottlenecks for large batches.

// Actually, for LeNet-5 with batch_size=4096, the bottleneck is likely the fully connected layers (16*5*5 -> 120).
// Let's focus on optimizing the Linear layers with a fused Matmul+ReLU kernel.

// Fused Matmul + ReLU Kernel
__global__ void matmul_relu_kernel(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ C, 
    int M, // Batch size
    int N, // Output features
    int K  // Input features
) {
    // Each thread block handles a row of the output matrix (one sample in batch)
    // Or we can use a tiled approach. For simplicity and speed with large M, 
    // let's assign one thread per element in C, but that's inefficient for GEMM.
    // Better: One block per row of C? No, too many blocks.
    // Standard GEMM structure:
    
    int batch_idx = blockIdx.y; // Batch index
    if (batch_idx >= M) return;

    int out_col = threadIdx.x + blockIdx.x * blockDim.x; // Output feature index
    
    if (out_col >= N) return;

    const float* A_row = A + batch_idx * K;
    const float* B_col = B + out_col * K; // B is stored as [K, N], so column major access or row major?
    // PyTorch Linear: out = xA^T + b. A is [out_features, in_features]. 
    // So B in kernel should be the weight matrix transposed for efficient memory access if we want col-major, 
    // or we just iterate K.
    
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A_row[k] * B_col[k];
    }
    
    // Apply ReLU
    if (sum > 0.0f) {
        C[batch_idx * N + out_col] = sum;
    } else {
        C[batch_idx * N + out_col] = 0.0f;
    }
}

// Fused Linear (Matmul + Bias + ReLU) Kernel
__global__ void linear_relu_kernel(
    const float* __restrict__ A, 
    const float* __restrict__ B, // Weights [Out, In]
    const float* __restrict__ bias, // Bias [Out]
    float* __restrict__ C, 
    int M, // Batch size
    int N, // Output features
    int K  // Input features
) {
    int batch_idx = blockIdx.y;
    if (batch_idx >= M) return;

    int out_col = threadIdx.x + blockIdx.x * blockDim.x;
    if (out_col >= N) return;

    const float* A_row = A + batch_idx * K;
    const float* B_col = B + out_col * K; 

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A_row[k] * B_col[k];
    }
    
    // Add bias and apply ReLU
    sum += bias[out_col];
    
    if (sum > 0.0f) {
        C[batch_idx * N + out_col] = sum;
    } else {
        C[batch_idx * N + out_col] = 0.0f;
    }
}

// Final Linear Layer (Matmul + Bias, no ReLU)
__global__ void linear_kernel(
    const float* __restrict__ A, 
    const float* __restrict__ B, // Weights [Out, In]
    const float* __restrict__ bias, // Bias [Out]
    float* __restrict__ C, 
    int M, // Batch size
    int N, // Output features
    int K  // Input features
) {
    int batch_idx = blockIdx.y;
    if (batch_idx >= M) return;

    int out_col = threadIdx.x + blockIdx.x * blockDim.x;
    if (out_col >= N) return;

    const float* A_row = A + batch_idx * K;
    const float* B_col = B + out_col * K; 

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A_row[k] * B_col[k];
    }
    
    C[batch_idx * N + out_col] = sum + bias[out_col];
}

// Python bindings
torch::Tensor linear_relu_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor bias) {
    // a: [M, K], b: [N, K], bias: [N]
    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(0);
    
    auto c = torch::empty({M, N}, a.options());
    
    const int block_size = 256;
    dim3 grid((N + block_size - 1) / block_size, M);
    
    linear_relu_kernel<<<grid, block_size>>>(
        a.data_ptr<float>(), 
        b.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        c.data_ptr<float>(), 
        M, N, K
    );
    
    CUDA_CHECK(cudaGetLastError());
    return c;
}

torch::Tensor linear_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor bias) {
    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(0);
    
    auto c = torch::empty({M, N}, a.options());
    
    const int block_size = 256;
    dim3 grid((N + block_size - 1) / block_size, M);
    
    linear_kernel<<<grid, block_size>>>(
        a.data_ptr<float>(), 
        b.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        c.data_ptr<float>(), 
        M, N, K
    );
    
    CUDA_CHECK(cudaGetLastError());
    return c;
}

"""

custom_cpp_source = (
    "torch::Tensor linear_relu_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor bias);"
    "torch::Tensor linear_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor bias);"
);

# Load the custom CUDA extensions
custom_ops = load_inline(
    name="custom_lenet_ops",
    cpp_sources=custom_cpp_source,
    cuda_sources=custom_cuda_source,
    functions=["linear_relu_cuda", "linear_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"]
)


class ModelNew(nn.Module):
    def __init__(self, num_classes):
        """
        Optimized LeNet-5 architecture using custom CUDA operators for Fully Connected layers.
        
        :param num_classes: The number of output classes.
        """
        super(ModelNew, self).__init__()
        
        # Convolutional layers remain unchanged as PyTorch's conv2d is highly optimized 
        # and the bottleneck here is the FC layers with large batch size.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        
        # Fully connected layers parameters stored as buffers or just used directly
        # We will use the custom CUDA operators for FC1, FC2, and FC3
        
        self.num_classes = num_classes

    def forward(self, x):
        """
        Forward pass of the optimized LeNet-5 model.
        
        :param x: The input tensor, shape (batch_size, 1, 32, 32)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        # First convolutional layer with ReLU activation and max pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Second convolutional layer with ReLU activation and max pooling
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Flatten the output for the fully connected layers
        # Shape: (batch_size, 16*5*5) -> (batch_size, 400)
        x = x.view(x.size(0), -1)
        
        # Get weights and biases from the original model's state dict if we were loading it, 
        # but here we assume the model is initialized with standard nn.Linear layers to get the weights.
        # However, since we are defining ModelNew from scratch, we need to access the weights.
        # To make this self-contained and functional without external state loading logic in forward,
        # we will define the FC layers as nn.Linear but then use their parameters in the custom kernel calls.
        
        # Note: In a real scenario, you might replace the nn.Linear modules with just buffers for weights/biases.
        # Here, we keep them to initialize weights properly if this class is instantiated and trained, 
        # but we override the forward pass logic.
        
        # FC1: 400 -> 120
        w_fc1 = self.fc1.weight.data.t() # Transpose because our kernel expects [Out, In] and PyTorch Linear stores [Out, In] but computes x @ W.T
        # Wait, PyTorch Linear: out = input @ weight.T + bias. 
        # Our kernel: sum(A_row[k] * B_col[k]). If A is [M, K] and B is [N, K], then B_col[k] is B[k][out_idx].
        # So if we pass weight.T to the kernel as B, it matches.
        # Let's stick to the kernel signature: linear_relu_cuda(A, B, bias) where A=[M,K], B=[N,K].
        # PyTorch FC1 weight is [120, 400]. So B should be weight.T -> [400, 120]? No.
        # Kernel logic: sum_{k=0}^{K-1} A[m,k] * B[n,k]. This computes (A @ B^T)[m,n].
        # We want out = x @ W.T + b. So we need B^T = W.T => B = W.
        # So if W is [Out, In], then B should be [Out, In].
        # Let's re-verify kernel:
        # A_row[k] is A[m, k]. B_col[k] is B[n, k].
        # Sum is dot product of row m of A and row n of B.
        # This is equivalent to (A @ B^T)[m, n].
        # We want x @ W.T. So we need B^T = W.T => B = W.
        # So we pass the weight matrix directly as B.
        
        # However, my kernel implementation above:
        # const float* B_col = B + out_col * K; 
        # This assumes B is stored in row-major order where each row is a feature?
        # If B is [N, K], then B[out_col] is the row for output feature 'out_col'.
        # B_col[k] accesses element k of that row.
        # So yes, if we pass Weight (shape [Out, In]) as B, it works.
        
        # FC1
        x = custom_ops.linear_relu_cuda(x, self.fc1.weight.data, self.fc1.bias.data)
        
        # FC2: 120 -> 84
        x = custom_ops.linear_relu_cuda(x, self.fc2.weight.data, self.fc2.bias.data)
        
        # FC3: 84 -> num_classes (No ReLU)
        x = custom_ops.linear_cuda(x, self.fc3.weight.data, self.fc3.bias.data)
        
        return x

# We need to define the fc layers inside __init__ so they exist for weight access
# But wait, if I define them as nn.Linear, they are part of the module.
# The code above references self.fc1 etc. So I must define them.

class ModelNew(nn.Module):
    def __init__(self, num_classes):
        """
        Optimized LeNet-5 architecture using custom CUDA operators for Fully Connected layers.
        
        :param num_classes: The number of output classes.
        """
        super(ModelNew, self).__init__()
        
        # Convolutional layers remain unchanged
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        """
        Forward pass of the optimized LeNet-5 model.
        
        :param x: The input tensor, shape (batch_size, 1, 32, 32)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        # First convolutional layer with ReLU activation and max pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Second convolutional layer with ReLU activation and max pooling
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)
        
        # Use custom CUDA operators for Fully Connected layers with fused ReLU
        x = custom_ops.linear_relu_cuda(x, self.fc1.weight.data, self.fc1.bias.data)
        x = custom_ops.linear_relu_cuda(x, self.fc2.weight.data, self.fc2.bias.data)
        
        # Final Linear Layer without ReLU
        x = custom_ops.linear_cuda(x, self.fc3.weight.data, self.fc3.bias.data)
        
        return x