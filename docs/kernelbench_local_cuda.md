# KernelBench Local CUDA Runbook

This note records the working command for generating and evaluating CUDA kernels
with a local OpenAI-compatible model server.

For the scoring and timing logic, see:

```text
docs/kernelbench_eval_logic.md
```

For the problem set structure, see:

```text
docs/kernelbench_problems.md
```

## Working Directory

Run all commands from the KernelBench repository:

```bash
cd /workspace/external/KernelBench
```

## Model Server

The local model server is:

```text
http://192.168.1.78:3132/v1
```

The detected model id is:

```text
gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf
```

You can verify it with:

```bash
curl -sS http://192.168.1.78:3132/v1/models
```

## CUDA Environment

This container does not expose CUDA under `/usr/local/cuda`. The working CUDA
toolkit path is:

```text
/workspace/.venv/lib/python3.11/site-packages/nvidia/cu13
```

KernelBench also needs `.venv/bin` in `PATH` so PyTorch can find `ninja`.

## Recommended Command

Use this exact command. Keep each environment assignment on one physical line,
or copy the whole block exactly.

```bash
PATH=/workspace/external/KernelBench/.venv/bin:/workspace/.venv/lib/python3.11/site-packages/nvidia/cu13/bin:$PATH \
CUDA_HOME=/workspace/.venv/lib/python3.11/site-packages/nvidia/cu13 \
LD_LIBRARY_PATH=/workspace/.venv/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-} \
.venv/bin/python scripts/generate_and_eval_single_sample.py \
  dataset_src=local \
  level=1 \
  problem_id=1 \
  server_type=local \
  model_name=gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf \
  server_address=192.168.1.78 \
  server_port=3132 \
  backend=cuda \
  gpu_arch=Ada \
  precision=fp32 \
  temperature=0.0 \
  prompt_option=one_shot \
  log_generated_kernel=True
```

Single-line version:

```bash
PATH=/workspace/external/KernelBench/.venv/bin:/workspace/.venv/lib/python3.11/site-packages/nvidia/cu13/bin:$PATH CUDA_HOME=/workspace/.venv/lib/python3.11/site-packages/nvidia/cu13 LD_LIBRARY_PATH=/workspace/.venv/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-} .venv/bin/python scripts/generate_and_eval_single_sample.py dataset_src=local level=1 problem_id=1 server_type=local model_name=gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf server_address=192.168.1.78 server_port=3132 backend=cuda gpu_arch=Ada precision=fp32 temperature=0.0 prompt_option=one_shot log_generated_kernel=True
```

## Output Files

Generated CUDA/Python wrapper:

```text
results/eval_logs/generated_kernel_level_1_problem_1.py
```

Raw model response before code extraction:

```text
results/eval_logs/raw_generation_level_1_problem_1.txt
```

View the generated kernel:

```bash
sed -n '1,260p' results/eval_logs/generated_kernel_level_1_problem_1.py
```

## Expected Successful Result

A successful generation and evaluation should include:

```text
compiled=True
correctness=True
correctness_trials=(5 / 5)
```

The generated kernel may still be slower than the PyTorch baseline. For example,
the first working CUDA generation for level 1 problem 1 was correct but slower:

```text
runtime=22.0 ms
ref_runtime=2.38 ms
```

## Common Errors

### Broken PATH or LD_LIBRARY_PATH line

Error:

```text
bash: python3.11/site-packages/nvidia/cu13/bin:...: No such file or directory
bash: lib:...: No such file or directory
```

Cause: the shell command was split in the middle of a path, for example:

```bash
PATH=/workspace/external/KernelBench/.venv/bin:/workspace/.venv/lib/
python3.11/site-packages/nvidia/cu13/bin:$PATH
```

Fix: do not break a path across lines. Use the recommended command above.

### Missing ninja

Error:

```text
Ninja is required to load C++ extensions
```

Fix: make sure `.venv/bin` is first in `PATH`:

```bash
PATH=/workspace/external/KernelBench/.venv/bin:$PATH
```

### Missing CUDA_HOME

Error:

```text
CUDA_HOME environment variable is not set
```

Fix:

```bash
CUDA_HOME=/workspace/.venv/lib/python3.11/site-packages/nvidia/cu13
```

### Missing libcudart.so.13

Error:

```text
libcudart.so.13: cannot open shared object file
```

Fix:

```bash
LD_LIBRARY_PATH=/workspace/.venv/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
```

### Static check says missing __global__

Error:

```text
Static check failed ... Missing __global__ kernel definition
```

Cause: the model generated a PyTorch fallback or the code extractor selected the
wrong code block.

Fixes:

1. Use `backend=cuda`, not `backend=triton`.
2. Use `temperature=0.0`.
3. Keep `log_generated_kernel=True` and inspect both output files.
