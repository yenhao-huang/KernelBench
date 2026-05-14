# Codex GPT-5.5 CUDA KernelBench Retime Score

Score record for retiming existing Codex GPT-5.5 kernels on the 206-problem KernelBench leaderboard dataset. No code generation was run.

## Run

```text
model: Codex GPT-5.5
backend: cuda
precision: fp32
source_run: runs/ralph_codex_cuda_all
retime_runs: ['runs/codex_gpt55_retime_l1', 'runs/codex_gpt55_retime_l2', 'runs/codex_gpt55_retime_l3', 'runs/codex_gpt55_retime_l4']
leaderboard_dataset_total: 206
levels: [1, 2, 3, 4]
measure_performance: True
num_perf_trials: 100
```

Source artifacts:

```text
source kernels: runs/ralph_codex_cuda_all/generated_kernels/
L1: runs/codex_gpt55_retime_l1/eval_results.json
L2: runs/codex_gpt55_retime_l2/eval_results.json
L3: runs/codex_gpt55_retime_l3/eval_results.json
L4: runs/codex_gpt55_retime_l4/eval_results.json
log: runs/codex_gpt55_retime_logs/retime.log
```

## Overall Score

```text
correct: 142 / 206
accuracy: 68.93%
evaluated: 206 / 206
compiled: 190 / 206
compile_rate: 92.23%
```

Status counts:

```text
correct: 142
output_mismatch: 46
runtime_error: 15
compile_failed: 2
shape_mismatch: 1
```

## By Level

| Level | Total | Evaluated | Compiled | Correct | Accuracy | Compile Rate |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 54 | 54 | 51 | 42 | 77.78% | 94.44% |
| 2 | 95 | 95 | 93 | 73 | 76.84% | 97.89% |
| 3 | 48 | 48 | 46 | 27 | 56.25% | 95.83% |
| 4 | 9 | 9 | 0 | 0 | 0.00% | 0.00% |

## Speed

Speedup is `reference_runtime / generated_kernel_runtime`, parsed from the retime eval log.

| Scope | Kernels | Geomean Speedup | Median Speedup | Faster Than Reference |
|---|---:|---:|---:|---:|
| All | 142 | 0.3631x | 0.5860x | 42 / 142 |
| L1 | 42 | 0.3019x | 0.3336x | 8 / 42 |
| L2 | 73 | 0.4066x | 0.8374x | 29 / 73 |
| L3 | 27 | 0.3564x | 0.6903x | 5 / 27 |
| L4 | 0 | 0.0000x | 0.0000x | 0 / 0 |

FastP over the 206-problem denominator:

| Threshold | Count | Score |
|---:|---:|---:|
| >1.0x | 42 / 206 | 20.39% |
| >1.5x | 18 / 206 | 8.74% |
| >2.0x | 8 / 206 | 3.88% |

## Top Speedups

| Problem | Speedup | Ref Runtime | Kernel Runtime |
|---|---:|---:|---:|
| L2 P80 | 1089.219x | 2.93 | 0.00269 |
| L2 P42 | 16.512x | 21.3 | 1.29 |
| L2 P83 | 15.145x | 7.3 | 0.482 |
| L1 P88 | 8.885x | 5.26 | 0.592 |
| L2 P51 | 5.627x | 6.19 | 1.1 |
| L1 P94 | 3.032x | 27.8 | 9.17 |
| L2 P18 | 2.883x | 2.97 | 1.03 |
| L2 P92 | 2.311x | 11.0 | 4.76 |
| L2 P7 | 1.909x | 17.2 | 9.01 |
| L2 P50 | 1.862x | 22.9 | 12.3 |

## Slowest Correct Kernels

| Problem | Speedup | Ref Runtime | Kernel Runtime |
|---|---:|---:|---:|
| L2 P55 | 0.010x | 5.97 | 573.0 |
| L2 P56 | 0.010x | 5.97 | 573.0 |
| L2 P59 | 0.015x | 5.95 | 406.0 |
| L3 P38 | 0.016x | 26.3 | 1660.0 |
| L3 P36 | 0.017x | 10.1 | 583.0 |
| L2 P36 | 0.017x | 4.3 | 248.0 |
| L2 P45 | 0.023x | 9.7 | 430.0 |
| L1 P71 | 0.030x | 1.91 | 62.8 |
| L2 P23 | 0.031x | 2.35 | 77.0 |
| L1 P64 | 0.034x | 11.1 | 327.0 |

## Per-Level Histograms

- L1: correct 42, output_mismatch 9, runtime_error 3
- L2: correct 73, output_mismatch 19, compile_failed 1, runtime_error 1, shape_mismatch 1
- L3: correct 27, output_mismatch 18, runtime_error 2, compile_failed 1
- L4: runtime_error 9

## Notes

- This score uses existing Codex kernels only; no code generation was run.
- Kernel files were symlinked into runs/codex_gpt55_retime_l{1,2,3,4} using evaluator-compatible filenames.
- Accuracy changed from the earlier leaderboard record 144/206 to 142/206 in this retime run.
