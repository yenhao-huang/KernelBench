# KernelBench Problem Guide

This document explains what the local KernelBench problems are, how they are
organized, and how to read one problem file.

## Problem Counts

The local repository contains 270 Python problem files:

```text
level 1: 100
level 2: 100
level 3: 50
level 4: 20
total: 270
```

The files live under:

```text
KernelBench/level1/
KernelBench/level2/
KernelBench/level3/
KernelBench/level4/
```

## Levels

### Level 1: Single-Kernel Operators

Count:

```text
100 problems
```

These are foundational tensor operators. The task is usually to replace one
PyTorch operation with one custom kernel.

Examples:

```text
1_Square_matrix_multiplication_
2_Standard_matrix_multiplication_
3_Batched_matrix_multiplication
4_Matrix_vector_multiplication_
19_ReLU
23_Softmax
33_BatchNorm
40_LayerNorm
47_Sum_reduction_over_a_dimension
54_conv_standard_3D__square_input__square_kernel
```

Typical optimization targets:

```text
matmul
reductions
activations
normalization
convolution
pooling
loss functions
```

Level 1 is the best place to debug generation, compilation, correctness, and
basic performance measurement.

### Level 2: Simple Fusion Patterns

Count:

```text
100 problems
```

These combine multiple operators. The goal is often to fuse several PyTorch
operations into fewer custom kernels.

Examples:

```text
1_Conv2D_ReLU_BiasAdd
2_ConvTranspose2d_BiasAdd_Clamp_Scaling_Clamp_Divide
3_ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU
7_Conv3d_ReLU_LeakyReLU_GELU_Sigmoid_BiasAdd
9_Matmul_Subtract_Multiply_ReLU
12_Gemm_Multiply_LeakyReLU
```

Typical optimization targets:

```text
Conv + activation
Conv + bias + clamp
Matmul/GEMM + elementwise ops
normalization + activation
pooling + reductions
```

Level 2 is where kernel fusion starts to matter. A correct but unfused solution
can still be slow.

### Level 3: Full Model Architectures

Count:

```text
50 problems
```

These are complete neural network blocks or model architectures. The task is to
optimize the whole model or selected bottlenecks inside it.

Examples:

```text
1_MLP
4_LeNet5
5_AlexNet
7_GoogleNetInceptionV1
8_ResNetBasicBlock
9_ResNet18
10_ResNet101
11_VGG16
12_VGG19
20_MobileNetV2
28_VisionTransformer
44_MiniGPTBlock
```

Typical optimization targets:

```text
large matmul layers
convolution blocks
attention blocks
recurrent layers
model-specific bottlenecks
```

Level 3 is harder because replacing every PyTorch op is usually unrealistic.
Good solutions often focus on the expensive parts and leave non-bottleneck
structure in PyTorch.

### Level 4: HuggingFace Model Architectures

Count:

```text
20 problems
```

These represent model workloads from HuggingFace-style architectures and
batch/sequence configurations.

Examples:

```text
1_EleutherAI-gpt-neo-2p7B_bs32_seq256
2_facebook-opt-1p3b_bs1_seq2047
5_google-bigbird-roberta-base_bs1_seq4095
6_facebook-bart-large_bs1_seq1023
7_gpt2_bs32_seq256
10_google-bigbird-roberta-base_bs1024_seq32
11_google-electra-small-discriminator_bs1_seq511
```

Typical optimization targets:

```text
transformer blocks
attention
MLP/feed-forward layers
batch/sequence-shape-specific bottlenecks
```

Level 4 is the largest-granularity task set and is not the right starting point
for debugging local CUDA generation.

## Problem IDs

`problem_id` is the integer prefix in the filename.

Example:

```text
KernelBench/level1/1_Square_matrix_multiplication_.py
```

Use:

```text
level=1
problem_id=1
```

Another example:

```text
KernelBench/level2/9_Matmul_Subtract_Multiply_ReLU.py
```

Use:

```text
level=2
problem_id=9
```

Problem IDs are loaded by parsing the filename prefix before the first
underscore. They are 1-indexed logical IDs, not zero-indexed array positions.

## Problem File Interface

Each benchmark problem is a Python file that defines at least:

```python
class Model(nn.Module):
    ...

def get_inputs():
    ...

def get_init_inputs():
    ...
```

`Model` is the PyTorch reference implementation.

`get_init_inputs()` returns constructor inputs for `Model`.

`get_inputs()` returns forward-pass inputs used during correctness and timing.

For generated kernels, the model output must define:

```python
class ModelNew(nn.Module):
    ...
```

The evaluator compares:

```text
Model(*get_init_inputs())(*get_inputs())
```

against:

```text
ModelNew(*get_init_inputs())(*get_inputs())
```

## Example: Level 1 Problem 1

File:

```text
KernelBench/level1/1_Square_matrix_multiplication_.py
```

Reference operation:

```python
return torch.matmul(A, B)
```

Input shape:

```text
A: [4096, 4096]
B: [4096, 4096]
```

This is a square matrix multiplication benchmark. A custom CUDA solution must
compute the same output as PyTorch matmul.

Important caveat: PyTorch matmul usually calls cuBLAS. A simple hand-written
tiled CUDA kernel can be correct but slower than the reference.

## How To Inspect Problems

List problem counts:

```bash
for d in KernelBench/level*; do
  printf '%s ' "$d"
  find "$d" -maxdepth 1 -type f -name '*.py' | wc -l
done
```

List problem names in a level:

```bash
find KernelBench/level1 -maxdepth 1 -type f -name '*.py' \
  | sed 's#.*/##; s#\.py$##' \
  | sort -V
```

Open a specific problem:

```bash
sed -n '1,220p' KernelBench/level1/1_Square_matrix_multiplication_.py
```

Use the dataset loader:

```bash
.venv/bin/python - <<'PY'
from kernelbench.dataset import construct_kernelbench_dataset

for level in [1, 2, 3, 4]:
    dataset = construct_kernelbench_dataset(level=level, source='local')
    print(f'level {level}: {len(dataset)}')
PY
```

## Recommended Debug Order

For local CUDA generation, debug in this order:

```text
level 1 problem 1    # square matmul, simple but cuBLAS baseline is strong
level 1 reductions   # sum/mean/max/min reductions
level 1 activations  # ReLU, GELU, Softmax variants
level 2 fusion       # Conv/GEMM plus elementwise chains
level 3 models       # only after generation and eval are stable
level 4 HF models    # last
```

Start with Level 1 because failures are easier to classify:

```text
bad code extraction
static checker failure
compilation failure
runtime error
output mismatch
correct but slow
```

## Related Docs

Local CUDA command and environment:

```text
docs/kernelbench_local_cuda.md
```

Evaluation logic:

```text
docs/kernelbench_eval_logic.md
```
