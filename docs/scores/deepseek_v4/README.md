# Deepseek V4 Pro CUDA KernelBench Score

Score record for Deepseek V4 Pro on the 206-problem KernelBench leaderboard dataset.

## Run

```text
model: openrouter/deepseek/deepseek-v4-pro
backend: cuda
precision: fp32
prompt_option: one_shot
temperature: 0.0
max_tokens: 8192
num_samples: 1
leaderboard_dataset_total: 206
levels: [1, 2, 3, 4]
```

Source artifacts:

```text
L1: runs/openrouter_deepseek_v4_pro_leaderboard_l1/eval_results.json
L2: runs/openrouter_deepseek_v4_pro_leaderboard_l2/eval_results.json
L3: runs/openrouter_deepseek_v4_pro_leaderboard_l3/eval_results.json
L4: runs/openrouter_deepseek_v4_pro_leaderboard_l4/eval_results.json
log: runs/openrouter_deepseek_v4_pro_leaderboard_logs/leaderboard_run.log
log: runs/openrouter_deepseek_v4_pro_leaderboard_logs/leaderboard_l4_local_run.log
```

## Overall Score

```text
correct: 94 / 206
accuracy: 45.63%
evaluated: 178 / 206
compiled: 145 / 206
compile_rate: 70.39%
```

Status counts:

```text
correct: 94
output_mismatch: 32
runtime_error: 24
compile_failed: 21
generation_error: 21
shape_mismatch: 7
static_failed: 7
```

## By Level

| Level | Total | Evaluated | Compiled | Correct | Accuracy | Compile Rate |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 54 | 46 | 34 | 24 | 44.44% | 62.96% |
| 2 | 95 | 91 | 83 | 59 | 62.11% | 87.37% |
| 3 | 48 | 37 | 28 | 11 | 22.92% | 58.33% |
| 4 | 9 | 4 | 0 | 0 | 0.00% | 0.00% |

## Speed

Speedup is `reference_runtime / generated_kernel_runtime`. These values use same-run reference runtimes parsed from the eval logs because `eval_results.json` does not persist `ref_runtime`.

| Scope | Kernels | Geomean Speedup | Median Speedup | Faster Than Reference |
|---|---:|---:|---:|---:|
| All | 89 | 0.6802x | 1.0107x | 50 / 89 |
| L1 | 19 | 0.3500x | 0.3306x | 4 / 19 |
| L2 | 59 | 0.8275x | 1.0545x | 40 / 59 |
| L3 | 11 | 0.7488x | 1.0035x | 6 / 11 |
| L4 | 0 | 0.0000x | 0.0000x | 0 / 0 |

FastP over the 206-problem denominator:

| Threshold | Count | Score |
|---:|---:|---:|
| >1.0x | 50 / 206 | 24.27% |
| >1.5x | 13 / 206 | 6.31% |
| >2.0x | 6 / 206 | 2.91% |

## Top Speedups

| Problem | Speedup | Ref Runtime | Kernel Runtime |
|---|---:|---:|---:|
| L2 P23 | 523.179x | 2.37 | 0.00453 |
| L2 P18 | 27.248x | 2.97 | 0.109 |
| L1 P88 | 17.032x | 10.1 | 0.593 |
| L2 P82 | 2.392x | 23.9 | 9.99 |
| L2 P57 | 2.299x | 8.07 | 3.51 |
| L3 P50 | 2.096x | 10.9 | 5.2 |
| L1 P40 | 1.973x | 10.1 | 5.12 |
| L2 P87 | 1.752x | 21.2 | 12.1 |
| L2 P16 | 1.718x | 34.7 | 20.2 |
| L2 P35 | 1.650x | 14.5 | 8.79 |

## Slowest Correct Kernels

| Problem | Speedup | Ref Runtime | Kernel Runtime |
|---|---:|---:|---:|
| L2 P53 | 0.012x | 6.21 | 520.0 |
| L2 P63 | 0.012x | 2.97 | 245.0 |
| L2 P12 | 0.012x | 2.97 | 242.0 |
| L2 P9 | 0.027x | 2.99 | 112.0 |
| L2 P29 | 0.027x | 2.98 | 111.0 |
| L2 P99 | 0.038x | 5.39 | 142.0 |
| L1 P74 | 0.040x | 4.76 | 118.0 |
| L2 P59 | 0.041x | 5.92 | 143.0 |
| L1 P81 | 0.051x | 6.1 | 119.0 |
| L1 P10 | 0.066x | 0.978 | 14.9 |

## Per-Level Histograms

- L1: correct 24, runtime_error 10, compile_failed 7, generation_error 7, output_mismatch 5, static_failed 1
- L2: correct 59, output_mismatch 16, compile_failed 6, shape_mismatch 6, generation_error 4, runtime_error 4
- L3: correct 11, output_mismatch 11, compile_failed 8, generation_error 6, runtime_error 6, static_failed 5, shape_mismatch 1
- L4: generation_error 4, runtime_error 4, static_failed 1

## Classification Notes

- Kernel-not-found eval records are classified as generation_error unless generation logs show a static check failure.
- Some L1 and L2 Kernel-not-found records are stale eval results from an earlier generation failure; later retries produced kernel files, but eval skipped them because eval_results.json already contained entries.

## Notes

- L4 generation used the local dataset because the HuggingFace loader in this workspace only supports levels 1-3.
- `benchmark_eval_analysis.py` is an offline analyzer; it does not regenerate kernels or rerun CUDA evaluation.
