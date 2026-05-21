import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernels for Conv2d (im2col + gemm) and ReLU fusion
# We will implement a fused conv2d+relu kernel to reduce memory bandwidth overhead.
# For simplicity and robustness in this inline context, we will use cuBLAS for the GEMM part 
# after im2col, but fuse the activation.

custom_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

// Helper to check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s at line %d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Kernel for ReLU activation
__global__ void relu_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        output[idx] = val > 0.0f ? val : 0.0f;
    }
}

// Kernel for im2col transformation
__global__ void im2col_kernel(const float* data_im, const int height, const int width, 
                              const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
                              const int stride_h, const int stride_w,
                              float* data_col) {
    // Each thread computes one element of the output column matrix
    // Output shape: (kernel_h * kernel_w * in_channels, out_height * out_width)
    
    const int out_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    const int out_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    const int dim_col = kernel_h * kernel_w * in_channels; // This assumes in_channels is known or passed
    
    // We need to know in_channels. Let's pass it as a parameter.
    // Actually, let's make the kernel generic by passing dimensions.
}

// Optimized im2col that handles arbitrary input channels
__global__ void im2col_kernel_general(const float* data_im, const int height, const int width, 
                                      const int in_channels, const int kernel_h, const int kernel_w, 
                                      const int pad_h, const int pad_w, const int stride_h, const int stride_w,
                                      float* data_col) {
    // Thread index corresponds to a specific output position and channel group?
    // Standard approach: Each thread computes one element of the column matrix.
    // Total elements in col: (kernel_h * kernel_w * in_channels) * (out_h * out_w)
    
    int total_elements = kernel_h * kernel_w * in_channels * height * width; // Approximate bound, need exact output size
    
    // Let's calculate output dimensions inside or pass them.
    // To keep it simple, we assume the caller calculates grid/block sizes based on total output elements of col.
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // We don't have out_h/out_w directly here unless passed. 
    // Let's pass out_h and out_w.
}

// Better approach: Pass all dimensions
__global__ void im2col_kernel_v2(const float* data_im, const int height, const int width, 
                                 const int in_channels, const int kernel_h, const int kernel_w, 
                                 const int pad_h, const int pad_w, const int stride_h, const int stride_w,
                                 const int out_h, const int out_w,
                                 float* data_col) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Total number of elements in the column matrix
    int total_cols = kernel_h * kernel_w * in_channels;
    int total_rows = out_h * out_w;
    
    if (idx >= total_cols * total_rows) return;
    
    // Decompose idx into col_index and row_index
    // data_col is laid out as [col_index][row_index] or [row_index][col_index]?
    // cuBLAS expects column-major. 
    // Standard im2col usually produces a matrix where columns are the patches.
    // So shape is (kernel_h * kernel_w * in_channels, out_h * out_w).
    
    int col_idx = idx / total_rows; // Which patch element (0 to kh*kw*ic-1)
    int row_idx = idx % total_rows; // Which output position (0 to oh*ow-1)
    
    int out_y = row_idx / out_w;
    int out_x = row_idx % out_w;
    
    int kh_idx = col_idx / (in_channels * kernel_w);
    int kw_idx = (col_idx / in_channels) % kernel_w;
    int ic_idx = col_idx % in_channels;
    
    int in_y = out_y * stride_h - pad_h + kh_idx;
    int in_x = out_x * stride_w - pad_w + kw_idx;
    
    if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
        data_col[idx] = data_im[(ic_idx * height + in_y) * width + in_x];
    } else {
        data_col[idx] = 0.0f;
    }
}

// Fused Conv2d + ReLU using im2col and cuBLAS
torch::Tensor conv2d_relu_cuda(torch::Tensor input, torch::Tensor weight, 
                               int stride_h, int stride_w, int pad_h, int pad_w) {
    // Input: (N, C_in, H_in, W_in)
    // Weight: (C_out, C_in, K_h, K_w)
    
    auto N = input.size(0);
    auto C_in = input.size(1);
    auto H_in = input.size(2);
    auto W_in = input.size(3);
    
    auto C_out = weight.size(0);
    auto K_h = weight.size(2);
    auto K_w = weight.size(3);
    
    auto H_out = (H_in + 2 * pad_h - K_h) / stride_h + 1;
    auto W_out = (W_in + 2 * pad_w - K_w) / stride_w + 1;
    
    // Allocate output tensor
    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());
    
    // Allocate column buffer
    int col_size = K_h * K_w * C_in * H_out * W_out;
    auto col_buffer = torch::zeros({col_size}, input.options());
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Process each image in the batch
    for (int n = 0; n < N; ++n) {
        const float* input_ptr = input.data_ptr<float>() + n * C_in * H_in * W_in;
        float* col_ptr = col_buffer.data_ptr<float>();
        
        // Im2Col
        int block_size = 256;
        int num_blocks = (col_size + block_size - 1) / block_size;
        
        im2col_kernel_v2<<<num_blocks, block_size>>>(
            input_ptr, H_in, W_in, C_in, K_h, K_w, pad_h, pad_w, stride_h, stride_w, H_out, W_out, col_ptr
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // GEMM: output = weight * col
        // Weight shape: (C_out, C_in*K_h*K_w) -> flattened to 2D matrix for cuBLAS
        // Col shape: (C_in*K_h*K_w, H_out*W_out)
        // Output shape: (C_out, H_out*W_out)
        
        const float* weight_ptr = weight.data_ptr<float>();
        float* output_ptr = output.data_ptr<float>() + n * C_out * H_out * W_out;
        
        int m = C_out;
        int k = C_in * K_h * K_w;
        int n_cols = H_out * W_out;
        
        // cuBLAS is column-major. 
        // We want: Output(C_out, H*W) = Weight(C_out, K*C) * Col(K*C, H*W)
        // In column major, this is standard GEMM C = alpha*A*B + beta*C
        // A is (m x k), B is (k x n), C is (m x n)
        
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                    m, n_cols, k, 
                    &alpha, 
                    weight_ptr, m, 
                    col_ptr, k, 
                    &beta, 
                    output_ptr, m);
    }
    
    cublasDestroy(handle);
    
    // Apply ReLU in-place on the output tensor
    int total_elements = N * C_out * H_out * W_out;
    if (total_elements > 0) {
        int block_size = 256;
        int num_blocks = (total_elements + block_size - 1) / block_size;
        relu_kernel<<<num_blocks, block_size>>>(output.data_ptr<float>(), output.data_ptr<float>(), total_elements);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    return output;
}

// Custom MaxPool2d with stride and padding
__global__ void maxpool_kernel(const float* input, float* output, 
                               int height, int width, int channels,
                               int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w) {
    int out_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = channels * out_h * out_w;
    
    if (idx >= total_elements) return;
    
    int c = idx / (out_h * out_w);
    int rem = idx % (out_h * out_w);
    int oh = rem / out_w;
    int ow = rem % out_w;
    
    float max_val = -FLT_MAX;
    
    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int ih = oh * stride_h - pad_h + kh;
            int iw = ow * stride_w - pad_w + kw;
            
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                float val = input[(c * height + ih) * width + iw];
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
    }
    
    output[idx] = max_val;
}

torch::Tensor maxpool2d_cuda(torch::Tensor input, int kernel_h, int kernel_w, 
                             int stride_h, int stride_w, int pad_h, int pad_w) {
    auto N = input.size(0);
    auto C = input.size(1);
    auto H_in = input.size(2);
    auto W_in = input.size(3);
    
    auto H_out = (H_in + 2 * pad_h - kernel_h) / stride_h + 1;
    auto W_out = (W_in + 2 * pad_w - kernel_w) / stride_w + 1;
    
    auto output = torch::zeros({N, C, H_out, W_out}, input.options());
    
    int block_size = 256;
    // Process each channel and spatial position. 
    // Total elements per batch: C * H_out * W_out
    // We can launch one kernel for the whole batch or per batch. 
    // Let's do per batch to keep memory access simple, or just flatten everything.
    
    int total_elements = N * C * H_out * W_out;
    if (total_elements > 0) {
        int num_blocks = (total_elements + block_size - 1) / block_size;
        maxpool_kernel<<<num_blocks, block_size>>>(
            input.data_ptr<float>(), output.data_ptr<float>(),
            H_in, W_in, C,
            kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    return output;
}

// Custom AdaptiveAvgPool2d -> Global Avg Pool for (1,1) output
__global__ void global_avg_pool_kernel(const float* input, float* output, 
                                       int height, int width, int channels, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels;
    
    if (idx >= total_elements) return;
    
    int b = idx / channels;
    int c = idx % channels;
    
    float sum = 0.0f;
    int num_pixels = height * width;
    
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            sum += input[((b * channels + c) * height + h) * width + w];
        }
    }
    
    output[idx] = sum / num_pixels;
}

torch::Tensor adaptive_avg_pool2d_1x1_cuda(torch::Tensor input) {
    auto N = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);
    
    auto output = torch::zeros({N, C, 1, 1}, input.options());
    
    int total_elements = N * C;
    if (total_elements > 0) {
        int block_size = 256;
        int num_blocks = (total_elements + block_size - 1) / block_size;
        global_avg_pool_kernel<<<num_blocks, block_size>>>(
            input.data_ptr<float>(), output.data_ptr<float>(), H, W, C, N
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    return output;
}

// Custom Linear (FC) layer using cuBLAS
torch::Tensor linear_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    // Input: (N, in_features)
    // Weight: (out_features, in_features)
    // Bias: (out_features)
    
    auto N = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    
    auto output = torch::zeros({N, out_features}, input.options());
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // GEMM: Output(N, out) = Input(N, in) * Weight^T(out, in)
    // Or Output = Weight * Input^T?
    // PyTorch Linear: output = input @ weight.T + bias
    // So we want C = A * B where A is (N, in), B is (in, out).
    // cuBLAS column major: C(m,n) = A(m,k) * B(k,n)
    // Here m=N, k=in, n=out.
    // A is input (N, in). B is weight.T (in, out).
    
    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    
    // We need to transpose weight for cuBLAS if we want to use standard GEMM C=AB
    // Or we can compute C = A * B^T using cublasSgemm with transA, transB flags.
    // Let's use: Output(N, out) = Input(N, in) * Weight(in, out)^T ? No.
    // Standard Linear: y = x W^T + b.
    // So we want to multiply Input (N, in) by Weight.T (in, out).
    // Let A = Input (N, in). Let B = Weight (out, in).
    // We want C = A * B^T.
    // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, out, in, &alpha, input_ptr, N, weight_ptr, in, &beta, output_ptr, N);
    
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                N, out_features, in_features, 
                &alpha, 
                input_ptr, N, 
                weight_ptr, in_features, 
                &beta, 
                output_ptr, N);
                
    // Add bias
    if (bias.numel() > 0) {
        int block_size = 256;
        int num_blocks = (N * out_features + block_size - 1) / block_size;
        
        __global__ void add_bias_kernel(float* output, const float* bias, int N, int out_features) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int total = N * out_features;
            if (idx < total) {
                int b = idx / out_features;
                int f = idx % out_features;
                output[idx] += bias[f];
            }
        }
        
        add_bias_kernel<<<num_blocks, block_size>>>(output_ptr, bias.data_ptr<float>(), N, out_features);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    cublasDestroy(handle);
    return output;
}

// Flatten kernel
__global__ void flatten_kernel(const float* input, float* output, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = channels * height * width;
    
    if (idx >= total_elements) return;
    
    // Input is (N, C, H, W), we are processing one sample at a time or all?
    // Let's assume input is flattened from (C, H, W) to (C*H*W) for one sample.
    // The caller handles the batch loop or we pass N.
    // For simplicity in this inline example, let's assume we flatten the last 3 dims of a single tensor.
    
    int c = idx / (height * width);
    int rem = idx % (height * width);
    int h = rem / width;
    int w = rem % width;
    
    output[idx] = input[(c * height + h) * width + w];
}

torch::Tensor flatten_cuda(torch::Tensor input) {
    // Input: (N, C, H, W)
    // Output: (N, C*H*W)
    auto N = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);
    
    auto output = torch::zeros({N, C * H * W}, input.options());
    
    int block_size = 256;
    int total_elements = N * C * H * W;
    if (total_elements > 0) {
        int num_blocks = (total_elements + block_size - 1) / block_size;
        
        __global__ void flatten_batch_kernel(const float* input, float* output, int N, int C, int H, int W) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= N * C * H * W) return;
            
            int n = idx / (C * H * W);
            int rem = idx % (C * H * W);
            
            int c = rem / (H * W);
            int h_w_rem = rem % (H * W);
            int h = h_w_rem / W;
            int w = h_w_rem % W;
            
            output[idx] = input[((n * C + c) * H + h) * W + w];
        }
        
        flatten_batch_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), N, C, H, W);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    return output;
}

// Dropout kernel (simple version for inference or training with p=0.0 as per model)
// The model uses dropout(0.0), which is effectively identity. 
// However, to be safe and generic, we can implement a simple dropout or just skip if p=0.
// Since the model specifies 0.0, we can just return input. But let's provide a kernel for completeness if needed.
// For this optimization, since p=0.0, we will just pass through in Python logic or use a trivial kernel.

"""

custom_cpp_source = """
#include <torch/extension.h>

torch::Tensor conv2d_relu_cuda(torch::Tensor input, torch::Tensor weight, 
                               int stride_h, int stride_w, int pad_h, int pad_w);
torch::Tensor maxpool2d_cuda(torch::Tensor input, int kernel_h, int kernel_w, 
                             int stride_h, int stride_w, int pad_h, int pad_w);
torch::Tensor adaptive_avg_pool2d_1x1_cuda(torch::Tensor input);
torch::Tensor linear_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
torch::Tensor flatten_cuda(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_relu", &conv2d_relu_cuda, "Conv2d + ReLU CUDA");
    m.def("maxpool2d", &maxpool2d_cuda, "MaxPool2d CUDA");
    m.def("adaptive_avg_pool2d_1x1", &adaptive_avg_pool2d_1x1_cuda, "AdaptiveAvgPool2d 1x1 CUDA");
    m.def("linear", &linear_cuda, "Linear CUDA");
    m.def("flatten", &flatten_cuda, "Flatten CUDA");
}
"""

# Load the custom extension
custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=custom_cpp_source,
    cuda_sources=custom_cuda_source,
    functions=["conv2d_relu", "maxpool2d", "adaptive_avg_pool2d_1x1", "linear", "flatten"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcublas"]
)


class InceptionModuleNew(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        super(InceptionModuleNew, self).__init__()
        
        # 1x1 convolution branch: Conv2d + ReLU
        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
        
        # 3x3 convolution branch: Conv2d(1x1) + ReLU -> Conv2d(3x3) + ReLU
        # We can fuse the first conv+relu, but the second one is separate.
        # For simplicity in this inline example, we will use standard PyTorch for the inner convs 
        # if they are small, or custom if beneficial. Given the overhead of launching kernels,
        # fusing the entire inception branch might be complex. 
        # However, the prompt asks to replace operators. Let's replace the main Conv2d+ReLU patterns.
        
        self.branch3x3_1 = nn.Conv2d(in_channels, reduce_3x3, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)
        
        # 5x5 convolution branch
        self.branch5x5_1 = nn.Conv2d(in_channels, reduce_5x5, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)
        
        # Max pooling branch
        self.branch_pool_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool_conv = nn.Conv2d(in_channels, pool_proj, kernel_size=1)

    def forward(self, x):
        # Branch 1x1: Conv + ReLU
        branch1x1 = custom_ops.conv2d_relu(x, self.branch1x1.weight, 1, 1, 0, 0)
        
        # Branch 3x3: Conv(1x1) + ReLU -> Conv(3x3) + ReLU
        x_3x3_1 = custom_ops.conv2d_relu(x, self.branch3x3_1.weight, 1, 1, 0, 0)
        branch3x3 = custom_ops.conv2d_relu(x_3x3_1, self.branch3x3_2.weight, 1, 1, 1, 1)
        
        # Branch 5x5: Conv(1x1) + ReLU -> Conv(5x5) + ReLU
        x_5x5_1 = custom_ops.conv2d_relu(x, self.branch5x5_1.weight, 1, 1, 0, 0)
        branch5x5 = custom_ops.conv2d_relu(x_5x5_1, self.branch5x5_2.weight, 1, 1, 2, 2)
        
        # Branch Pool: MaxPool + Conv(1x1) + ReLU
        x_pool = custom_ops.maxpool2d(x, 3, 3, 1, 1, 1, 1)
        branch_pool = custom_ops.conv2d_relu(x_pool, self.branch_pool_conv.weight, 1, 1, 0, 0)
        
        # Concatenate
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        
        # Initial layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Inception Modules
        self.inception3a = InceptionModuleNew(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModuleNew(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.inception4a = InceptionModuleNew(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModuleNew(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModuleNew(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModuleNew(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModuleNew(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.inception5a = InceptionModuleNew(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModuleNew(832, 384, 192, 384, 48, 128, 128)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.0)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Conv1 + ReLU + MaxPool1
        x = custom_ops.conv2d_relu(x, self.conv1.weight, 2, 2, 3, 3)
        x = custom_ops.maxpool2d(x, 3, 3, 2, 2, 1, 1)
        
        # Conv2 + ReLU
        x = custom_ops.conv2d_relu(x, self.conv2.weight, 1, 1, 0, 0)
        
        # Conv3 + ReLU + MaxPool2
        x = custom_ops.conv2d_relu(x, self.conv3.weight, 1, 1, 1, 1)
        x = custom_ops.maxpool2d(x, 3, 3, 2, 2, 1, 1)
        
        # Inception Blocks
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = custom_ops.maxpool2d(x, 3, 3, 2, 2, 1, 1)
        
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = custom_ops.maxpool2d(x, 3, 3, 2, 2, 1, 1)
        
        x = self.inception5a(x)
        x = self.inception5b(x)
        
        # Adaptive Avg Pool (1x1)
        x = custom_ops.adaptive_avg_pool2d_1x1(x)
        
        # Flatten
        x = custom_ops.flatten(x)
        
        # Dropout (0.0 is identity)
        if self.dropout.p > 0:
            x = self.dropout(x)
            
        # Linear Layer
        x = custom_ops.linear(x, self.fc.weight, self.fc.bias)
        
        return x

# Test code
batch_size = 10
input_channels = 3
height = 224
width = 224
num_classes = 1000

def get_inputs():
    return [torch.rand(batch_size, input_channels, height, width)]

def get_init_inputs():
    return [num_classes]