# Evaluation Pipeline

This document records the current KernelBench evaluation pipeline used for
local model generation and Codex-based generation.

## Pipelines

There are two generation paths in this workspace:

```text
local OpenAI-compatible server  -> KernelBench eval
Codex CLI gateway               -> KernelBench eval
```

Both paths eventually produce the same artifact layout:

```text
runs/<run_name>/summary.json
runs/<run_name>/results.jsonl
runs/<run_name>/per_problem/
runs/<run_name>/raw_generation/
runs/<run_name>/generated_kernels/
```

## Shared Evaluation Steps

For each problem, the evaluator does:

```text
1. Load KernelBench problem.
2. Build backend-specific prompt, usually backend=cuda.
3. Query generator.
4. Save raw model response.
5. Extract generated Python/CUDA code.
6. Save extracted kernel.
7. Run static checker.
8. Compile generated ModelNew.
9. Compare ModelNew against reference Model.
10. Write per-problem JSON.
11. Update summary.json.
```

The generated answer must define:

```python
class ModelNew(nn.Module):
    ...
```

For CUDA, the generated code is expected to contain:

```text
__global__
load_inline
```

Correctness uses 5 random trials per problem in the current batch runs.
Performance timing is disabled for score/accuracy runs unless explicitly stated.

## Codex CLI Gateway

The Codex gateway is implemented inside:

```text
scripts/ralph_eval_gemma4_cuda_all.py
```

Use:

```text
--server-type codex_cli
```

The gateway calls:

```bash
codex exec --model gpt-5.5
```

The prompt is wrapped with hard requirements:

```text
- Return only one fenced python code block defining ModelNew.
- Write at least one explicit CUDA __global__ kernel in cuda_sources.
- Use torch.utils.cpp_extension.load_inline to compile it.
- Do not call torch.matmul, torch.mm, torch.bmm, torch.einsum, or torch.nn.functional compute ops in ModelNew.forward.
- Do not use cuBLAS, CUTLASS, Thrust, or other library matmul helpers.
- Do not add try/except fallback paths.
- Assume inputs are CUDA tensors with the dtype requested by the prompt.
```

The Codex subprocess uses read-only sandboxing:

```bash
codex exec \
  --sandbox read-only \
  --skip-git-repo-check \
  --cd /workspace/external/KernelBench \
  --model gpt-5.5 \
  --output-last-message <tmpfile> \
  -
```

The last Codex message is saved as the raw generation for that problem.

## Codex 10-Problem Ralph Run

Ralph config:

```text
/workspace/configs/ralph/kernelbench_codex_cuda_10.json
```

Run output:

```text
runs/ralph_codex_cuda_10_v2/
```

Ralph log:

```text
/workspace/logs/kernelbench/codex_cuda_10_ralph_v2.log
```

The run used:

```text
model: gpt-5.5 through codex exec
backend: cuda
precision: fp32
prompt_option: one_shot
correctness_trials_per_problem: 5
performance_timing: disabled
problem_scope: first 10 local KernelBench problems
```

Command embedded in the Ralph config:

```bash
PYTHONPATH=src \
PATH=/workspace/external/KernelBench/.venv/bin:/workspace/.venv/lib/python3.11/site-packages/nvidia/cu13/bin:$PATH \
CUDA_HOME=/workspace/.venv/lib/python3.11/site-packages/nvidia/cu13 \
LD_LIBRARY_PATH=/workspace/.venv/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-} \
CUDA_VISIBLE_DEVICES=${KERNELBENCH_CUDA_VISIBLE_DEVICES:-0} \
.venv/bin/python scripts/ralph_eval_gemma4_cuda_all.py \
  --output-dir runs/ralph_codex_cuda_10_v2 \
  --levels 1,2,3,4 \
  --limit 10 \
  --dataset-src local \
  --server-type codex_cli \
  --model-name codex-gpt-5.5 \
  --codex-model gpt-5.5 \
  --backend cuda \
  --gpu-arch Ada \
  --precision fp32 \
  --temperature 0.0 \
  --prompt-option one_shot \
  --num-correct-trials 5
```

Result:

```text
completed: 10 / 10
compiled: 10 / 10
correct: 7 / 10
accuracy_all: 70.00%
compile_rate_all: 100.00%
```

Per-problem result:

```text
1  Square matrix multiplication        correct
2  Standard matrix multiplication      correct
3  Batched matrix multiplication       correct
4  Matrix vector multiplication        correct
5  Matrix scalar multiplication        failed: CUDA OOM during correctness
6  Matmul large K                      correct
7  Matmul small K                      failed: CUDA OOM during correctness
8  Matmul irregular shapes             correct
9  Tall skinny matmul                  failed: CUDA OOM during correctness
10 3D tensor matrix multiplication     correct
```

Important interpretation:

```text
The three failed Codex problems did answer, did pass static check, and did compile.
They failed during KernelBench correctness evaluation because the output tensors
are large and torch.allclose needed additional GPU memory.
```

## Gemma4 Full Ralph Run

Ralph config:

```text
/workspace/configs/ralph/kernelbench_gemma4_cuda_all.json
```

Run output:

```text
runs/ralph_gemma4_cuda_all/
```

Score record:

```text
plans/scores/gemma/README.md
plans/scores/gemma/score_record.json
```

Result:

```text
completed: 270 / 270
correct: 32 / 270
accuracy_all: 11.85%
evaluated: 145 / 270
accuracy_evaluated: 22.07%
compiled: 68 / 270
compile_rate_all: 25.19%
```

Status counts:

```text
evaluated: 145
static_failed: 87
generation_error: 29
extraction_failed: 7
evaluation_error: 2
```

## OOM Notes

Some KernelBench Level 1 problems have very large outputs.

Example:

```text
problem 5: A shape = [65536, 16384]
one float32 tensor ~= 4 GiB
```

During correctness evaluation, memory can include:

```text
input tensor
reference output
ModelNew output
temporary tensors used by torch.allclose
```

Therefore, a generated kernel can compile and be structurally valid but still
fail with:

```text
torch.OutOfMemoryError
```

This should be treated as:

```text
evaluation_oom / inconclusive
```

not automatically as:

```text
model did not answer
```

## Useful Commands

Inspect a summary:

```bash
cat runs/ralph_codex_cuda_10_v2/summary.json
```

List failed Codex 10-problem results:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

root = Path('runs/ralph_codex_cuda_10_v2/per_problem')
for path in sorted(root.glob('level_*/problem_*.json')):
    data = json.loads(path.read_text())
    result = data.get('eval_result') or {}
    metadata = result.get('metadata') or {}
    if result.get('correctness') is not True:
        print(
            data['level'],
            data['problem_id'],
            data['problem_name'],
            data['status'],
            result.get('compiled'),
            result.get('correctness'),
            metadata.get('runtime_error_name') or metadata.get('correctness_issue')
        )
PY
```

Open a generated Codex kernel:

```bash
sed -n '1,220p' runs/ralph_codex_cuda_10_v2/generated_kernels/level_1/problem_5.py
```

Open a raw Codex answer:

```bash
sed -n '1,220p' runs/ralph_codex_cuda_10_v2/raw_generation/level_1/problem_5.txt
```

