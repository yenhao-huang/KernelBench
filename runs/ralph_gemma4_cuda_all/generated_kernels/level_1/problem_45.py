import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 2D Average Pooling
avg_pool2d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int in_h,
    int in_w,
    int kernel_size,
    int stride,
    int padding,
    int out_h,
    int out_w
) {
    // Calculate the index for the current output pixel
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_h_idx = out_idx / out_w;
            int out_w_idx = multi_dim_idx(multi_idx(weight-weight.size(0), weight.size(flat_idx-flat_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_keykey_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_floor(out_idx / out_w);
                        int out_w_idx = out_idx % out_w;

                        int b = out_idx / (channels * out_h * out_w);
                        int c = (out_idx / (out_h * out_w)) % channels;
xi = xi + dy[i] dy[key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_key_size_key_size_key_key_        
                        int b = out_idx / (channels * out_h * out_w）
                        int b = out_idx / (channels *out_h * out_w);
                        int_idx = int_idx + idx;
                }
            }
        }
    }
}
"""