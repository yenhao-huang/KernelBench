# OpenRouter Support

KernelBench uses LiteLLM for non-local LLM calls, so OpenRouter works through
the `openrouter/<provider>/<model>` model prefix.

## Setup

Set your OpenRouter key in the shell or in `.env`:

```bash
export OPENROUTER_API_KEY=sk-or-...
```

The same key is listed in `.env.example`:

```bash
OPENROUTER_API_KEY=sk-or-...
```

The single-problem script prints estimated token counts before and after the
OpenRouter call:

```text
[Token estimate] {'prompt_tokens_est': ..., 'raw_generation_tokens_est': ..., 'extracted_kernel_tokens_est': ...}
```

The final evaluation output also prints the generated kernel runtime compared
with the PyTorch reference:

```text
[Runtime comparison] kernel=... ms pytorch_ref=... ms speedup=...x slowdown=...x
```

## Single Problem

Generate and evaluate one kernel locally:

```bash
PATH=/workspace/external/KernelBench/.venv/bin:/workspace/.venv/lib/python3.11/site-packages/nvidia/cu13/bin:$PATH \
CUDA_HOME=/workspace/.venv/lib/python3.11/site-packages/nvidia/cu13 \
LD_LIBRARY_PATH=/workspace/.venv/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-} \
uv run python scripts/generate_and_eval_single_sample.py dataset_src=huggingface level=1 problem_id=1 eval_mode=local server_type=openrouter model_name=openrouter/qwen/qwen3-coder-next max_tokens=8192 temperature=0.0 gpu_arch=Ada
```

This requires a local GPU for evaluation. To only generate kernels in batch and
evaluate later, use the batch flow below.

## Batch Generation And Evaluation

Generate kernels for a full level:

```bash
uv run python scripts/generate_samples.py run_name=openrouter_level_1 dataset_src=huggingface level=1 server_type=openrouter model_name=openrouter/qwen/qwen3-coder num_workers=8 temperature=0.0
```

Evaluate the generated kernels:

```bash
PATH=/workspace/external/KernelBench/.venv/bin:/workspace/.venv/lib/python3.11/site-packages/nvidia/cu13/bin:$PATH \
CUDA_HOME=/workspace/.venv/lib/python3.11/site-packages/nvidia/cu13 \
LD_LIBRARY_PATH=/workspace/.venv/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-} \
uv run python scripts/eval_from_generations.py run_name=openrouter_level_1 dataset_src=local level=1 num_gpu_devices=1 timeout=300
```

Analyze the results:

```bash
uv run python scripts/benchmark_eval_analysis.py run_name=openrouter_level_1 level=1 hardware=L40S_matx3 baseline=baseline_time_torch
```

Adjust `hardware` and `baseline` to match your timing setup.

## Leaderboard Non-OOM Subset

The local leaderboard in `docs/leaderboard.md` compares models on the 206
problems where the Codex GPT-5.5 CUDA run did not hit OOM. To run OpenRouter on
the same subset, generate one level at a time with `problem_ids`.

Level 1:

```bash
uv run python scripts/generate_samples.py run_name=openrouter_leaderboard dataset_src=huggingface level=1 server_type=openrouter model_name=openrouter/qwen/qwen3-coder-next problem_ids=1,2,3,4,6,8,10,12,13,14,15,16,17,18,40,43,47,48,49,50,51,52,53,54,56,58,60,61,62,64,65,66,67,68,69,70,71,72,73,74,75,77,78,79,80,81,82,83,85,86,88,93,94,95 num_workers=8 temperature=0.0 max_tokens=8192
```

Level 2:

```bash
uv run python scripts/generate_samples.py run_name=openrouter_leaderboard dataset_src=huggingface level=2 server_type=openrouter model_name=openrouter/qwen/qwen3-coder-next problem_ids=5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99 num_workers=8 temperature=0.0 max_tokens=8192
```

Level 3:

```bash
uv run python scripts/generate_samples.py run_name=openrouter_leaderboard dataset_src=huggingface level=3 server_type=openrouter model_name=openrouter/qwen/qwen3-coder-next problem_ids=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,19,20,21,22,23,24,25,26,27,28,29,30,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50 num_workers=8 temperature=0.0 max_tokens=8192
```

Level 4:

```bash
uv run python scripts/generate_samples.py run_name=openrouter_leaderboard dataset_src=huggingface level=4 server_type=openrouter model_name=openrouter/qwen/qwen3-coder-next problem_ids=4,5,6,7,13,14,15,16,17 num_workers=4 temperature=0.0 max_tokens=8192
```

Then evaluate each level with the same IDs. Local evaluation in this workspace
needs the CUDA environment prefix:

```bash
PATH=/workspace/external/KernelBench/.venv/bin:/workspace/.venv/lib/python3.11/site-packages/nvidia/cu13/bin:$PATH \
CUDA_HOME=/workspace/.venv/lib/python3.11/site-packages/nvidia/cu13 \
LD_LIBRARY_PATH=/workspace/.venv/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-} \
uv run python scripts/eval_from_generations.py run_name=openrouter_leaderboard dataset_src=local level=1 problem_ids=1,2,3,4,6,8,10,12,13,14,15,16,17,18,40,43,47,48,49,50,51,52,53,54,56,58,60,61,62,64,65,66,67,68,69,70,71,72,73,74,75,77,78,79,80,81,82,83,85,86,88,93,94,95 num_gpu_devices=1 timeout=300
```

Repeat the eval command for levels 2, 3, and 4 with the matching `problem_ids`
lists above.

This subset avoids the OOM cases observed in the Codex leaderboard run. It does
not guarantee that every newly generated OpenRouter kernel is OOM-free, because
a generated implementation can still allocate extra tensors or launch an invalid
kernel.

## Model Names

Pass OpenRouter models using LiteLLM's OpenRouter prefix:

```bash
model_name=openrouter/qwen/qwen3-coder
model_name=openrouter/anthropic/claude-sonnet-4.5
model_name=openrouter/openai/gpt-5.2
```

OpenRouter model availability changes over time, so check the current model slug
on OpenRouter before launching a large run.

## Notes

- `server_type=openrouter` selects the OpenRouter preset in `kernelbench.utils`.
- `max_tokens` defaults to `8192` for the OpenRouter preset.
- KernelBench does not pass `top_k` to OpenRouter models because many
  OpenAI-compatible routes reject that parameter.
- Local CUDA evaluation in this workspace needs the `PATH`, `CUDA_HOME`, and
  `LD_LIBRARY_PATH` prefix shown above. Without it, PyTorch extension compilation
  fails because CUDA is not exposed under `/usr/local/cuda`.
- Generation consumes OpenRouter credits. Start with one problem before running
  a full level.
