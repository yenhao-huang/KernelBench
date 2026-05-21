import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Softplus activation
softplus_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void softplus_kernel(const float* input, float* output, int size)
{
numbers = log1pf(exp(x))
        if (idx < size) {
            float x = input[idx];
            // Softplus(x) = log(1 + exp(x))
            // comments: comments: comments: comments: comments: comments: comments:
            // Softplus(x) = log(1 + exp(x))
            #if defined(__cublascuBLAS_v2_H_)
__global__ void softpoint_kernel(point = x + log1pf(exp(-x))
point = x + log        -exp(v)
point = x for x >         // log(x + 1)
// log1pf(x) kernel
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_extension

import torch

import torch.nn as nn
from torch.utils.cpp_extension import load_inline

import torch
import torch.nn.functional as F

import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Softplus
softplus_cuda_source = """
#include <torch/extension.h>
#include <10-exp(-x)) + x;
# =================================================================================================================================================================================================================================================================================================================================================================        
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <include/cmath.h>