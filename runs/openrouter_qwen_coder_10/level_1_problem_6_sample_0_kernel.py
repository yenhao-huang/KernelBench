#include <torch/extension.h>
#include <cuda_runtime.h>

// Tiled GEMM kernel with shared memory optimization
template <int TILE_M, int TILE_N, int TILE_K>
__global__ void gemm_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    // Shared memory for tiles
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_N][TILE_K];
    
    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Global row and column indices for this thread
    int row = blockIdx.y * TILE_M + ty;
    int col = blockIdx.x * TILE_N + tx;
    
    // Accumulator for the result
    float acc = 0.0f;
    
    // Loop over tiles of K dimension
    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // Load tile of A into shared memory
        if (row < M && (k0 + tx) < K) {
            As[ty][tx] = A[row * K + (k0 + tx)];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Load tile of B into shared memory
        if (col < N && (k0 + ty) < K) {
            Bs[tx][ty] = B[(k0 + ty) * N + col];
        } else {
            Bs[tx][ty] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result to global memory
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// Specialized kernel for large K dimension with better tiling strategy
__global__ void gemm_large_k_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    // Use larger tile sizes for large K
    const int TILE_M = 16;
    const int TILE_N = 16;
    const int TILE_K = 256;
    
    // Shared memory for tiles
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_N][TILE_K];
    
    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Global row and column indices for this thread
    int row = blockIdx.y * TILE_M + ty;
    int col = blockIdx.x * TILE_N + tx;
    
    // Accumulator for the result
    float acc = 0.0f;
    
    // Loop over tiles of K dimension
    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // Load tile of A into shared memory with bounds checking
        if (row < M && (k0 + tx) < K) {
            As[ty][tx] = A[row * K + (k0 + tx)];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Load tile of B into shared memory with bounds checking
        if (col < N && (k0 + ty) < K) {
            Bs[tx][ty] = B[(k0 + ty) * N + col];
        } else {
            Bs[tx][ty] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result to global memory
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// Optimized kernel using warp-level operations for better performance
__global__ void gemm_warp_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    const int TILE_M = 32;
    const int TILE_N = 32;
    const int TILE_K = 32;
    
    // Shared memory
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_N][TILE_K];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int warp_id = threadIdx.y / 4 + (threadIdx.x / 32) * 8;
    int lane_id = threadIdx.x % 32;
    
    int row = blockIdx.y * TILE_M + ty;
    int col = blockIdx.x * TILE_N + tx;
    
    // Each warp computes a 4x4 tile
    float acc[4][4] = {{0}};
    
    // Loop over K dimension in chunks
    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // Load A tile
        if (row < M && (k0 + tx) < K) {
            As[ty][tx] = A[row * K + (k0 + tx)];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Load B tile
        if (col < N && (k0 + ty) < K) {
            Bs[ty][tx] = B[(k0 + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute with warp-level parallelism
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int row_idx = ty + i * 4;
                    int col_idx = tx + j * 4;
                    if (row_idx < TILE_M && col_idx < TILE_N) {
                        acc[i][j] += As[row_idx][k] * Bs[col_idx][k];
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results
    if (row < M && col < N) {
        C[row * N + col] = acc[ty % 4][tx % 4];
    }
}

// Best performing kernel for this specific problem size
__global__ void gemm_optimized_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    // Use 16x16 tiles with 256 K chunks for large K
    const int TILE_M = 16;
    const int TILE_N = 16;
    const int TILE_K = 256;
    
    // Shared memory
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_N][TILE_K];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = blockIdx.y * TILE_M + ty;
    int col = blockIdx.x * TILE_N + tx;
    
    // Accumulator
    float acc = 0.0f;
    
    // Loop over K dimension
    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // Load A tile
        if (row < M && (k0 + tx) < K) {
            As[ty][tx] = A[row * K + (k0 + tx)];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Load B tile
        if (col < N && (k0 + ty) < K) {
            Bs[tx][ty] = B[(k0 + ty) * N + col];
        } else {
            Bs[tx][ty] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute dot product
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

torch::Tensor gemm_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    // Create output tensor
    auto C = torch::empty({M, N}, A.options());
    
    // Configure kernel launch parameters
    const int TILE_M = 16;
    const int TILE_N = 16;
    
    dim3 threads(TILE_N, TILE_M);  // 16x16 threads per block
    dim3 blocks((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    
    // Launch optimized kernel
    gemm_optimized_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    
    return C;
}
"""

gemm_cpp_source = "torch::Tensor gemm_cuda(torch::Tensor A, torch::Tensor B);"

// Compile the inline CUDA code
gemm = load_inline(
    name="gemm",
    cpp_sources=gemm_cpp_source,
    cuda_sources=gemm_source,
    functions=["gemm_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gemm = gemm

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.gemm.gemm_cuda(A, B)