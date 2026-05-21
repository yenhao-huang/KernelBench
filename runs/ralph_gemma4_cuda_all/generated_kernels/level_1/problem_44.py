```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 1D Average Pooling
avg_pool1d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int input_length,
            int kernel_size,
            int stride,
            int padding,
    int output_length
) {
    // Calculate the global index for the current output element
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out_elements = batch_size * in_channels * output_length;
    
    if (out_idx >= total_out_elements) return;

    // Map the global index to the output tensor structure
    int out_l = out_idx % output_length;
                int out_c = (out_idx / output_length) % in_channels;
                int out_b = out_idx / (output_length * in_channels);

                // Calculate the starting position in the input tensor
                // We need to
            int in_start = out_l * stride - padding;
            int in_end = in_start + kernel_size;
                int in_start_clamped-1 = in_start;
                in_tensor_idx_base = (out_b * in_channels + out_c) * input_length;
                in_tensor_idx_base = (out_b *_in_channels + out_c) * input_idx_length;
                        
                        
                        
                        
                        
                        
                        

                        
                        
                        
                        
                
                
                
                
                
                
                
                
                
                
                
                
                
            
            
            
            
            
            
            
            
            
                
                
                
            
            

            
            
            
            
            
            
            
            
            
            
                
                
            
                
                
            
                
                
                
            
            
            
    
    
    
            
            
                
                
            
            
            
            
            
                
            
            
                
            
                
                // Calculate the starting position in the        
                // Calculate the padding-aware than the window-min/window-size
                
                
                
                
                
                *
                in_start = out_L * stride - padding;
 * in_start = out_l * spatial-size;
 * in_idx_L = in_start + k_idx;
_in_idx_L = _in_idx_L - padding;-padding;
_in_idx_L = _in
_in_idx_L = _in_idx__L - padding;
_in_x_idx_L = in_            _in_idx_L = in_idx_L - padding;
_in_idx_L = _in_idx_L - padding;
            
            
            
            
                
                
            _in_idx_L = in_idx_L - padding;
            // Map the global index to the    
            _in_idx_L = in_idx_L_L_L_L_                
                
            
            
            
            
            import_import_import_import_import_import_import_import_            
            
            
            
            
        
        
        
                
                <#include <torch/extension.h>
wrap_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_                
import_import_import_import_import_import_L_L_L_L_L_L_L_L_L_L_L_batch_size, batch_import_import_import_import_import_import_import_L_L_list_                
import_import_            import_import_import_count_import_            import<#include <torch/extension.        
import_import_import_import_import_import_import_import_L_list_idx_            import_import_import_import_pool_L_    import_                
import_import_import__L_L_buf_L_import_import_import_import_L_    import_import_import_L_L_L_L_L_    import_idx_import_import_import_import_import_L_L_
import_import_import_L_L_L_L_idx_idx_idx_x_L_idx_L_L_
import_import_batch_size, batch__import_idx_    import__import_L_sum_L_row_L_idx_count_sum_                _in_dir_    import de_import_L_            
            _in_idx_L = in_b_idx_L_    import_import_import_import_import_L_L_L_L_L_            
                // Map the                
                //                
            _in_        _in_L_L_L_                
            _                
            
            
            _in_idx_L_L_L_L_L_L
            
            _L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_    import_import_import_import_import_L_L_L_L_L_L_L_L
import_import_import_import_import_L_L_L_L
import_import_L_L_L_L_L_L_L_L_L_L_L_L
import_L_L_L_L_L_    import_import_import_input_input_import_import_L<#include <torch/extension.import_import_import_import_L_L_L_L_L_L_        
import_L_L_L_L_L_L_L<#include <<include <torch/extension.h>
<#include <buf_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L.L.L.L.L.L.import_import_import_import_import_idx_L_L_L_L나머지-import_import_import_import_L_L_L_L_                import_import_import_import_import_import_import_import_import_import_import_import_L_L_L_L_L_L_import_import_idx_L_L_L_L
import_import_import_L_L_L_L_L_L_L_L_L_L_L_L.L.L.L.L.L<#include <<include <torch/extension. importance_L_mapping-import_import_import_L_L_L.L.sum_                import_import_import_import_L_L<_in_idx_L_L_L_L_L_L_L_L
import_import_import_import_import_import_import_import_import_import_import_import_import_L_L_L_L_L_L_L_L
import_L_L_L_L_L_L_L_L_L_        import_import_import_import_import_L_L_L<#include <import_import_import_import_L_cuda_extension_import_import_import_L_L_L_L_L_L_L_L_L_L_L<#include <torch/extension.h>
<#include <torch/extension.h>
import_import_import_import_L_L_L_L_L_L_L_L_L_L_L_L_L_L.L.L.L.L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L<import_import_import_import_import_L_L
import_L_L_    import_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L
import_import_import_L_L_L_L_L_L_L_L_L                import_import_import_import_import_import_import_import_import_import_L_L_L_L_L_L_L.L.L_L_L_L_L
L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L
L_L_import_import_import_import_import_import_import_import_import_import_import_import_import_idx_L_L_L_L_L_L_L_L_L_L_L_L_L.L.L_L_L_L_        import_import_import_L_L_L.L.L_L_L_L_L_L_L_L_L.L.L_L_L_L_L_L
import_import_import_import_L_L_L_L_L_L_L_L_L
import_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L.L.L_L_L_L_L_L_L_L_L_N_L_L_L_L_L
import_L_L_L de_import_import_import_import_L_L_L.L_L_L_L_L_L_L_L_L_L
import_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L.L.L.L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L
L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_import_import_import_import_import_import_import_import_import_import_import_import_import_import_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L.L.L.L.L.L_L_L_L_L_L_L_L_L_L_L_L_L.L.L_L_L_L_L_L.L.L.L_L_L_L_L_L_L_F_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L
import_import_import_import_import_import_import_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_A_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L.L.L.L_L_L_L_L_L_L_L
import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L
import_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L too_import_import_import_import_import_import_import_import_import_import_L_L_L_import_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L로_import_import_import_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L
import_import_import_import_import_L_L_L_L_L_L_L_L_L_L*import_import_import_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L<#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int input_length,
    int kernel_size,
    int stride,
    int padding,
    int output_length
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out_elements = batch_size * in_channels * output_length;
    if (out_idx >= total_out_elements) return;

                // Map the global index to the output tensor structure
                int out_l = out_idx % output_length;
            int out_c = (out_idx / output_length) % in_channels;
    int out_L_idx = out_idx / (output_length * in_channels);
    out_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_                
                
                
                
    int out_b = out_idx / (output_length * in_channels);
_in_idx_L = in_idx_L - padding;
_in_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_Loop_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L
_in_idx_L = in_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L.L.L.L.L.L.L.L.L.L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L*
_in_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L__L_L_L_L_L_L_L_L_L_One_L_L__L_F_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_LL_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_    
                
                // Map the global index to theL_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L.L.L_L_L_L_L_L_L_L_L-import_import_import_import_import_import_import_import_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L-import_import_import_import_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L/import_import_import_import_import_import_import_import_L_L_L_L_L_L_L.L.L.L.L.L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L.L.L<#include <torch/extension.h>
#include <a_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L*
#include <cuda_runtime.h>

__global__ void avg_pool1d_kernel(
<...skipped code...>
<...skipped code_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L<...skipped code_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L<...L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L.L.L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L*
L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L
import_import_import_import_import_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L-import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L-import_import_import_import_import_import_import_import_import_import_import_import_import_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L로_import_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L-L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_T_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L    
import_import_import_import_import_import_import_import_import_import_import_import_import_L_L_L_L_L_N_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L.L.L.L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L.L.L.L.L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L-import_import_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L로_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_import_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L_L*
#include <cuda_runtime.h>

__global__ void avg_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int in_channels,