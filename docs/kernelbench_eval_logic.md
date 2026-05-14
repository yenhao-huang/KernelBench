# KernelBench Evaluation Logic

This document explains what happens after
`scripts/generate_and_eval_single_sample.py` receives a generated kernel.

The relevant entry point is:

```text
scripts/generate_and_eval_single_sample.py
```

The core evaluator is:

```text
src/kernelbench/eval.py
```

For the benchmark problem structure, see:

```text
docs/kernelbench_problems.md
```

## High-Level Flow

For a single sample run, KernelBench does this:

1. Load the reference problem from `KernelBench/level*/`.
2. Build a prompt for the selected backend, for example `backend=cuda`.
3. Query the model server.
4. Extract a code block from the model response.
5. Optionally save the raw response and extracted kernel.
6. Run static checks on the extracted kernel.
7. Compile and load `ModelNew`.
8. Compare `ModelNew` against the original PyTorch `Model`.
9. If correctness passes, benchmark `ModelNew`.
10. Benchmark the reference PyTorch `Model`.
11. Report correctness, runtime, reference runtime, and metadata.

## Static Check

Static checking happens before runtime evaluation when `check_kernel=True`.

For `backend=cuda`, the checker requires the generated code to contain:

```text
__global__
```

and one of:

```text
load_inline
cpp_extension
```

It also warns or fails on suspicious patterns such as PyTorch compute fallbacks,
for example:

```text
torch.matmul
torch.mm
torch.nn.functional.*
```

Common failure:

```text
Static check failed ... Missing __global__ kernel definition
```

That means the extracted code does not look like a real CUDA kernel.

## Code Extraction Failure

If the run fails with:

```text
AssertionError: Custom cuda kernel code generation failed
```

the evaluator did not reach static checking yet. It means KernelBench could not
extract a usable code block from the model response.

When `log_generated_kernel=True`, inspect:

```text
results/eval_logs/raw_generation_level_<level>_problem_<problem_id>.txt
```

Common causes:

1. The model returned text without a markdown code block.
2. The model emitted a partial Python/CUDA file but never defined `ModelNew`.
3. The model degenerated into repeated tokens before finishing the code.
4. The extractor selected the wrong code fence.

The local script has been adjusted to save the raw model response before
extraction and to fall back to raw Python-looking output when no markdown fence
is present. If the fallback output is malformed, the next failure should occur
in static checking or compilation, which is easier to diagnose.

## Compilation

For CUDA backend, the generated Python code is executed directly. The generated
code usually calls `torch.utils.cpp_extension.load_inline(...)`, which invokes
`ninja` and `nvcc` to compile the inline CUDA/C++ extension.

If this stage succeeds, the result can report:

```text
compiled=True
```

If compilation fails, correctness and performance are skipped. Example failures:

```text
Ninja is required to load C++ extensions
CUDA_HOME environment variable is not set
libcudart.so.13: cannot open shared object file
```

## Model Loading

KernelBench loads two models:

```text
Model     # original PyTorch reference model
ModelNew  # generated model from the LLM
```

The reference model and inputs come from the benchmark problem file. For example:

```text
KernelBench/level1/1_Square_matrix_multiplication_.py
```

Each problem defines:

```python
class Model(nn.Module):
    ...

def get_inputs():
    ...

def get_init_inputs():
    ...
```

The generated code must define:

```python
class ModelNew(nn.Module):
    ...
```

## Correctness Check

Correctness uses multiple random trials. In
`generate_and_eval_single_sample.py`, the current settings are:

```text
num_correct_trials=5
```

For each trial, KernelBench:

1. Generates a deterministic random seed from the base seed.
2. Calls the problem's `get_inputs()`.
3. Moves inputs to the selected CUDA device and dtype.
4. Runs the original PyTorch `Model`.
5. Runs the generated `ModelNew`.
6. Checks output shape.
7. Checks output values with `torch.allclose`.

For `precision=fp32`, tolerance is:

```text
atol=1e-4
rtol=1e-4
```

For `precision=fp16` or `precision=bf16`, tolerance is:

```text
atol=1e-2
rtol=1e-2
```

All correctness trials must pass. Example successful output:

```text
correctness=True
correctness_trials=(5 / 5)
```

If a trial fails, metadata records the maximum and average output difference:

```text
max_difference=[...]
avg_difference=[...]
correctness_issue=Output mismatch
```

## Performance Measurement

Performance is measured only if correctness passes.

For the single-sample script, the current settings are:

```text
measure_performance=True
num_perf_trials=100
timing_method=cuda_event
```

With `timing_method=cuda_event`, KernelBench:

1. Runs 3 warmup iterations.
2. Clears PyTorch's CUDA cache.
3. For each timing trial:
   - synchronizes CUDA,
   - clears L2 cache with a large dummy tensor,
   - records a CUDA start event,
   - runs the model,
   - records a CUDA end event,
   - synchronizes CUDA,
   - records elapsed time in milliseconds.
4. Reports mean, std, min, max, and trial count.

Example generated-kernel timing:

```text
runtime=22.0
runtime_stats={'mean': 22.0, 'std': 0.287, 'min': 21.9, 'max': 23.3, 'num_trials': 100}
```

The unit is milliseconds.

## Reference Runtime

After timing `ModelNew`, KernelBench also times the original PyTorch `Model`
with the same timing method and same number of trials.

Example:

```text
ref_runtime=2.38
ref_runtime_stats={'mean': 2.38, 'std': 0.0328, 'min': 2.32, 'max': 2.43, 'num_trials': 100}
```

This is the PyTorch baseline runtime in milliseconds.

For matrix multiplication, the PyTorch reference typically calls cuBLAS, so a
naive generated CUDA kernel can be correct but much slower.

## Effective Speedup

KernelBench computes:

```text
effective_speedup = ref_runtime / runtime
```

Example:

```text
ref_runtime = 2.38 ms
runtime = 22.0 ms
effective_speedup = 2.38 / 22.0 = 0.11x
```

Interpretation:

```text
> 1.0x  generated kernel is faster than reference
= 1.0x  same speed
< 1.0x  generated kernel is slower than reference
```

KernelBench also warns about suspiciously large speedups. The default threshold
inside `eval_kernel_against_ref` is:

```text
excessive_speedup_threshold=10
```

If a generated kernel appears more than 10x faster than the reference, metadata
is marked with:

```text
excessive_speedup=True
```

This is a reward-hacking guard, not proof of correctness by itself.

## Important Output Fields

```text
compiled
```

Whether `ModelNew` compiled and loaded successfully.

```text
correctness
```

Whether every correctness trial matched the reference output within tolerance.

```text
runtime
```

Mean generated-kernel runtime in milliseconds. Only meaningful when
`correctness=True`.

```text
ref_runtime
```

Mean PyTorch reference runtime in milliseconds.

```text
metadata
```

Extra details, including hardware, device, correctness trial count, runtime
errors, compilation errors, and output mismatch diagnostics.

## Where Outputs Are Saved

When `log_generated_kernel=True`, the extracted kernel is saved to:

```text
results/eval_logs/generated_kernel_level_<level>_problem_<problem_id>.py
```

The raw model response is saved to:

```text
results/eval_logs/raw_generation_level_<level>_problem_<problem_id>.txt
```

For level 1 problem 1:

```text
results/eval_logs/generated_kernel_level_1_problem_1.py
results/eval_logs/raw_generation_level_1_problem_1.txt
```
