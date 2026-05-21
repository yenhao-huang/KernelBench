---
name: codebook
description: Use for KernelBench runbook commands in this repository, including generating model kernels, evaluating generated kernels, and analyzing accuracy or speed from existing eval results.
---

# KernelBench Codebook

Use this skill when working in `/workspace/external/KernelBench` and the user asks for the command to generate kernels, evaluate kernels, or summarize benchmark accuracy/speed.

## Generate Kernel Code

Use `scripts/generate_samples.py`. For Deepseek V4 Pro L4, use the local dataset because the HuggingFace loader in this workspace only supports levels 1-3.

```bash
uv run python scripts/generate_samples.py \
  run_name=openrouter_deepseek_v4_pro_leaderboard_l4 \
  dataset_src=local \
  level=4 \
  server_type=openrouter \
  model_name=openrouter/deepseek/deepseek-v4-pro \
  problem_ids=4,5,6,7,13,14,15,16,17 \
  num_workers=2 \
  temperature=0.0 \
  max_tokens=8192
```

Generated kernels are written to:

```text
runs/<run_name>/level_<level>_problem_<id>_sample_0_kernel.py
```

## Evaluate Generated Kernels

Use `scripts/eval_from_generations.py` after generation.

```bash
uv run python scripts/eval_from_generations.py \
  run_name=<run_name> \
  dataset_src=local \
  level=<level> \
  problem_ids=<comma_separated_problem_ids> \
  num_gpu_devices=1 \
  timeout=300 \
  measure_performance=True
```

If CUDA paths are needed in this workspace, set:

```bash
export PATH="/workspace/external/KernelBench/.venv/bin:/workspace/.venv/lib/python3.11/site-packages/nvidia/cu13/bin:${PATH}"
export CUDA_HOME="/workspace/.venv/lib/python3.11/site-packages/nvidia/cu13"
export LD_LIBRARY_PATH="/workspace/.venv/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
```

## Analyze Accuracy And Speed

Use `scripts/benchmark_eval_analysis.py` for offline analysis. It reads an existing `eval_results.json` and a baseline timing JSON; it does not regenerate kernels or rerun CUDA eval.

```bash
uv run python scripts/benchmark_eval_analysis.py \
  run_name=<run_name> \
  level=<level> \
  hardware=<hardware> \
  baseline=<baseline>
```

Metrics include compile rate, correctness rate, geometric mean speedup for correct samples, and Fast_p scores.

## Deepseek Leaderboard Script

The bundled Deepseek leaderboard script is:

```bash
bash scripts/run_openrouter_deepseek_v4_pro_leaderboard.sh
```

In this repository version, L4 generation should use `dataset_src=local`; L1-L3 can use `dataset_src=huggingface`.
