import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA source code for fused convolution+ReLU (using cuDNN) and fused linear+ReLU (custom tiled GEMM)
custom_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cudnn.h>

// --------------------- Fused Conv2d + ReLU (cuDNN) ---------------------
torch::Tensor fused_conv_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups) {
    
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int K = weight.size(0); // out channels
    int R = weight.size(2); // kernel height
    int S = weight.size(3); // kernel width
    
    int out_h = (H + 2*padding_h - dilation_h*(R-1) - 1) / stride_h + 1;
    int out_w = (W + 2*padding_w - dilation_w*(S-1) - 1) / stride_w + 1;
    
    auto output = torch::zeros({N, K, out_h, out_w}, input.options());
    
    cudnnHandle_t handle;
    cudnnCreate(&handle);
    
    cudnnTensorDescriptor_t input_desc;
    cudnnCreateTensorDescriptor(&input_desc);
    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W);
    
    cudnnTensorDescriptor_t output_desc;
    cudnnCreateTensorDescriptor(&output_desc);
    cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, K, out_h, out_w);
    
    cudnnFilterDescriptor_t filter_desc;
    cudnnCreateFilterDescriptor(&filter_desc);
    cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, K, C/groups, R, S);
    
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnSetConvolution2dDescriptor(conv_desc, padding_h, padding_w, stride_h, stride_w, dilation_h, dilation_w, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    if (groups > 1) {
        cudnnSetConvolutionGroupCount(conv_desc, groups);
    }
    
    cudnnActivationDescriptor_t act_desc;
    cudnnCreateActivationDescriptor(&act_desc);
    cudnnSetActivationDescriptor(act_desc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0);
    
    cudnnConvolutionFwdAlgo_t algo;
    cudnnGetConvolutionForwardAlgorithm(handle, input_desc, filter_desc, conv_desc, output_desc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo);
    
    size_t workspace_size = 0;
    cudnnGetConvolutionForwardWorkspaceSize(handle, input_desc, filter_desc, conv_desc, output_desc, algo, &workspace_size);
    
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }
    
    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionBiasActivationForward(
        handle,
        &alpha,
        input_desc, input.data_ptr<float>(),
        filter_desc, weight.data_ptr<float>(),
        conv_desc, algo, workspace, workspace_size,
        &beta,
        output_desc, output.data_ptr<float>(),
        bias.data_ptr<float>(),
        act_desc,
        output_desc, output.data_ptr<float>()
    );
    
    if (workspace) cudaFree(workspace);
    cudnnDestroyActivationDescriptor(act_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroy(handle);
    
    return output;
}

// --------------------- Fused Linear + ReLU (custom tiled GEMM) ---------------------
#define BLOCK_M 16
#define BLOCK_N 16
#define BLOCK_K 16

__global__ void linear_relu_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K) {
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row_start = by * BLOCK_M;
    int col_start = bx * BLOCK_N;
    
    __shared__ float As[BLOCK_M][BLOCK_K];
    __shared__ float Bs[BLOCK_K][BLOCK_N];
    
    float sum = 0.0f;
    
    for (int k = 0; k < K; k += BLOCK_K) {
        if (row_start + ty < M && k + tx < K) {
            As[ty][tx] = A[(row_start + ty) * K + (k + tx)];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (col_start + tx < N && k + ty < K) {
            Bs[ty][tx] = B[(col_start + tx) * K + (k + ty)];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int i = 0; i < BLOCK_K; ++i) {
            sum += As[ty][i] * Bs[i][tx];
        }
        
        __syncthreads();
    }
    
    int row = row_start + ty;
    int col = col_start + tx;
    if (row < M && col < N) {
        float val = sum + bias[col];
        C[row * N + col] = (val > 0.0f) ? val : 0.0f;
    }
}

torch::Tensor fused_linear_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {
    
    int M = input.size(0);
    int K = input.size(1);
    int N = weight.size(0);
    
    auto output = torch::zeros({M, N}, input.options());
    
    dim3 block(BLOCK_N, BLOCK_M);
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    
    linear_relu_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K
    );
    
    return output;
}
"""

custom_ops_cpp_source = """
torch::Tensor fused_conv_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups);

torch::Tensor fused_linear_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias);
"""

# Compile the inline CUDA code
custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=custom_ops_cpp_source,
    cuda_sources=custom_ops_source,
    functions=["fused_conv_relu_cuda", "fused_linear_relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=["-lcudnn"],
)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        
        # Convolutional layers weights and biases (registered as parameters)
        self.conv1_weight = nn.Parameter(torch.empty(96, 3, 11, 11))
        self.conv1_bias = nn.Parameter(torch.empty(96))
        self.conv2_weight = nn.Parameter(torch.empty(256, 96, 5, 5))
        self.conv2_bias = nn.Parameter(torch.empty(256))
        self.conv3_weight = nn.Parameter(torch.empty(384, 256, 3, 3))
        self.conv3_bias = nn.Parameter(torch.empty(384))
        self.conv4_weight = nn.Parameter(torch.empty(384, 384, 3, 3))
        self.conv4_bias = nn.Parameter(torch.empty(384))
        self.conv5_weight = nn.Parameter(torch.empty(256, 384, 3, 3))
        self.conv5_bias = nn.Parameter(torch.empty(256))
        
        # Fully connected layers weights and biases
        self.fc1_weight = nn.Parameter(torch.empty(4096, 256 * 6 * 6))
        self.fc1_bias = nn.Parameter(torch.empty(4096))
        self.fc2_weight = nn.Parameter(torch.empty(4096, 4096))
        self.fc2_bias = nn.Parameter(torch.empty(4096))
        self.fc3 = nn.Linear(4096, num_classes)
        
        # Max pooling layers (no parameters)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(p=0.0)
        self.dropout2 = nn.Dropout(p=0.0)
        
        # Custom fused operators
        self.fused_conv_relu = custom_ops.fused_conv_relu_cuda
        self.fused_linear_relu = custom_ops.fused_linear_relu_cuda
        
        # Initialize weights (using default PyTorch initialization for consistency)
        self._initialize_weights()
    
    def _initialize_weights(self):
        # Initialize conv weights with Kaiming uniform, biases with zeros
        for m in [self.conv1_weight, self.conv2_weight, self.conv3_weight, self.conv4_weight, self.conv5_weight]:
            nn.init.kaiming_uniform_(m, a=0, mode='fan_in', nonlinearity='relu')
        for m in [self.conv1_bias, self.conv2_bias, self.conv3_bias, self.conv4_bias, self.conv5_bias]:
            nn.init.zeros_(m)
        nn.init.kaiming_uniform_(self.fc1_weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc1_bias)
        nn.init.kaiming_uniform_(self.fc2_weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc2_bias)
        # fc3 is initialized by its own constructor
    
    def forward(self, x):
        # Block 1: conv1 + relu + maxpool
        x = self.fused_conv_relu(x, self.conv1_weight, self.conv1_bias, 4, 4, 2, 2, 1, 1, 1)
        x = self.maxpool1(x)
        
        # Block 2: conv2 + relu + maxpool
        x = self.fused_conv_relu(x, self.conv2_weight, self.conv2_bias, 1, 1, 2, 2, 1, 1, 1)
        x = self.maxpool2(x)
        
        # Block 3: conv3 + relu
        x = self.fused_conv_relu(x, self.conv3_weight, self.conv3_bias, 1, 1, 1, 1, 1, 1, 1)
        
        # Block 4: conv4 + relu
        x = self.fused_conv_relu(x, self.conv4_weight, self.conv4_bias, 1, 1, 1, 1, 1, 1, 1)
        
        # Block 5: conv5 + relu + maxpool
        x = self.fused_conv_relu(x, self.conv5_weight, self.conv5_bias, 1, 1, 1, 1, 1, 1, 1)
        x = self.maxpool3(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Block 6: fc1 + relu + dropout
        x = self.fused_linear_relu(x, self.fc1_weight, self.fc1_bias)
        x = self.dropout1(x)
        
        # Block 7: fc2 + relu + dropout
        x = self.fused_linear_relu(x, self.fc2_weight, self.fc2_bias)
        x = self.dropout2(x)
        
        # Block 8: fc3 (no activation)
        x = self.fc3(x)
        
        return x