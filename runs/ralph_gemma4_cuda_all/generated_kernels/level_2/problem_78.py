import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for a fused reduction: MaxPool3d(2) -> MaxPool3d(3) -> Sum(dim=1)
# This kernel reduces memory bandwidth by performing the pooling and the channel-wise sum 
# in a single pass over the data, avoiding multiple intermediate large 3D tensors.
fused_pool_sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_pool_sum_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w) 
{
    // Each thread handles one (batch, spatial_idx) element in the output
    // The output shape is (batch, 1, out_d, out_h, out_w)
    int b = blockIdx.z;
    int d = blockIdx.y * blockDim.y + threadIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int w = threadIdx.z; // Using threadIdx.z for width to maximize occupancy

    // We want to map (b, d, h, w) to a single index in the output
    // But we need to iterate over channels to perform the sum.
    // To keep it simple and efficient, we'll use a 3D grid for (d, h, w) and loop over channels.
    
    // Re-calculating indices for a 3D grid approach
    // Grid: (out_h, out_d, batch)
    // Block: (out_w, out_h, out_d) - actually let's use a simpler mapping
}

// A more robust implementation:
// Each thread handles one (batch, spatial_idx) and sums across all channels.
// Output spatial dimensions: 
// After MaxPool3d(2): d1 = floor((in_d)/2), h1 = floor((in_h)/2), w1 = floor((in_w)/2)
// After MaxPool3d(3): d2 = floor((d1)/3), h2 = floor((h1)/3), w2 = floor((w1)/3)

__global__ void fused_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int B, int C, int D, int H, int W,
    int D2, int H2, int W2) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_spatial = D2 * H2 * W2;
    int total_elements = B * total_spatial;

    if (idx < total_elements) {
        int b = idx / total_spatial;
        int rem = idx % total_spatial;
        int d2 = rem / (H2 * W2);
        int rem2 = rem % (H2 * W2);
        int h2 = rem2 / W2;
        int w2 = rem2 % W2;

        // Map output (d2, h2, w2) back to input (d, h, w) through the two pooling layers
        // MaxPool1 (k=2, s=2): input_idx = d2*2 + offset
        // MaxPool2 (k=3, s=3): input_idx = d2*3 + offset
        // Combined: input_idx = (d2*3 + offset_in_p2)*2 + offset_in_p1
        
        float sum_val = 0.0f;
        for (int c = 0; c < C; ++c) {
            float max_val = -1e38f; // Approximate -infinity
            
            // We need to find the max in the window defined by the two pooling operations
            // The effective window size is (2*3) = 6
            // The effective stride is (2*3) = 6
            // However, the pooling layers are sequential.
            // Let's compute the window for the first pool, then for each element in that window, 
            // compute the window for the second pool.
            
            // To simplify: The combined operation is a MaxPool with kernel 6 and stride 6? 
            // No, because the second pool is applied to the result of the first.
            // Let's just simulate the two steps for the specific window.
            
            // For a fixed (d2, h2, w2), the window in the first pool (d1, h1, w1) is:
            // d1 in [d2*3, d2*3 + 2], h1 in [h2*3, h2*3 + 2], w1 in [w2*3, w2*3 + 2]
            // For each (d1, h1, w1), the window in the original input (d, h, w) is:
            // d in [d1*2, d1*2 + 1], h in [h1*2, h1*2 + 1], w in [w1*2, w1*2 + 1]

            for (int p2_d = 0; p2_d < 3; ++p2_d) {
                for (int p2_h = 0; p2_h < 3; ++p2_h) {
                    for (int p2_w = 0; p2_w < 3; ++p2_w) {
                        int d1 = d2 * 3 + p2_d;
                        int h1 = h2 * 3 + p2_h;
                        int w1 = w2 * 3 + p2_w;
                        
                        // Boundary check for first pool
                        if (d1 < (D/2) && h1 < (H/2) && w1 < (W/2)) {
                            // Find max in the 2x2x2 window of the first pool
                            float local_max = -1e38f;
                            for (int p1_d = 0; p1_d < 2; ++p1_d) {
                                for (int p1_h = 0; p1_h < 2; ++p1_h) {
                                    for (int p1_w = 0; p1_w < 2; ++p1_w) {
                                        int d = d1 * 2 + p1_d;
                                        int h = h1 * 2 + p1_h;
                                        int w = w1 * 2 + p1_w;
                                        if (d < D && h < H && w < W) {
                                            float val = input[((b * C + c) * D + d) * H * W + h * W + w];
                                            if (val > local_max) local_max = val;
                                        }
                                    }
                                }
                            }
                            if (local_max > max_val) max_val = local_max;
                        }
                    }
                }
            }
            sum_val += max_val;
        }
        output[b * total_spatial + rem] = sum_val;
    }
}

torch::Tensor fused_pool_sum_cuda(torch::Tensor input) {
    auto B = input.size(0);
    auto C = input.size(1);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);

    int D1 = D / 2;
    int H1 = H / 2;
    int W1 = W / 2;

    int D2 = D1 / 3;
    int H2 = H1 / 3;
    int W2 = W1 / 3;

    auto output = torch::zeros({B, 1, D2, H2, W2}, input.options());

    int total_spatial = D2 * H2 * W2;
    int total_elements = B * total_spatial;

    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, D, H, W,
        D2, H2, W2
    );

    return output;
}
"""

fused_pool_sum_cpp_source = "torch::Tensor fused_pool_sum_cuda(torch::Tensor input);"

fused_pool_sum = load_inline(
    name="fused_pool_sum",
    cpp_sources=fused_pool_sum_cpp_source,
    cuda_sources=fused_pool_sum_source,
    functions=["fused_pool_sum_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized Model using a custom CUDA kernel that fuses MaxPool3d(2), 
    MaxPool3d(3), and the channel-wise sum into a single operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.fused_op = fused_pool_sum.fused_pool_sum_cuda

    def forward(self, x):
        # ConvTranspose3d is highly optimized in cuDNN, so we keep it as is.
        # The subsequent pooling and sum operations are fused to save memory bandwidth.
        x = self.conv_transpose(x)
        x = self.fused_op(x)
        return x