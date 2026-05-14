# Qwen3.6 35B A3B KernelBench Retime Score

Existing generated kernels retimed on the 206-problem leaderboard dataset. No code generation was run.

## Overall Score

```text
correct: 36 / 206
accuracy: 17.48%
evaluated: 206 / 206
compiled: 106 / 206
compile_rate: 51.46%
```

Status counts:

```text
compile_failed: 84
correct: 36
output_mismatch: 51
runtime_error: 28
shape_mismatch: 7
```

## By Level

| Level | Total | Evaluated | Compiled | Correct | Accuracy | Compile Rate |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 54 | 54 | 28 | 12 | 22.22% | 51.85% |
| 2 | 95 | 95 | 62 | 23 | 24.21% | 65.26% |
| 3 | 48 | 48 | 16 | 1 | 2.08% | 33.33% |
| 4 | 9 | 9 | 0 | 0 | 0.00% | 0.00% |

## Speed

Speedup is `reference_runtime / generated_kernel_runtime`, parsed from retime eval logs.

| Scope | Kernels | Geomean Speedup | Median Speedup | Faster Than Reference |
|---|---:|---:|---:|---:|
| All | 36 | 0.437x | 0.946x | 16 / 36 |
| L1 | 12 | 0.229x | 0.139x | 2 / 12 |
| L2 | 23 | 0.601x | 1.044x | 14 / 23 |
| L3 | 1 | 0.681x | 0.681x | 0 / 1 |
| L4 | 0 | 0.000x | 0.000x | 0 / 0 |

| Threshold | Count | Score |
|---:|---:|---:|
| >1.0x | 16 / 206 | 7.77% |
| >1.5x | 7 / 206 | 3.40% |
| >2.0x | 2 / 206 | 0.97% |

## Top Speedups

| Problem | Speedup | Ref Runtime | Kernel Runtime |
|---|---:|---:|---:|
| L1 P88 | 8.900x | 5.26 | 0.591 |
| L2 P57 | 2.720x | 8.05 | 2.96 |
| L2 P7 | 1.909x | 17.2 | 9.01 |
| L2 P46 | 1.767x | 15.6 | 8.83 |
| L2 P48 | 1.765x | 6.6 | 3.74 |
| L2 P93 | 1.747x | 17 | 9.73 |
| L2 P16 | 1.689x | 34.8 | 20.6 |
| L2 P31 | 1.463x | 14.1 | 9.64 |
| L2 P58 | 1.347x | 27.2 | 20.2 |
| L1 P43 | 1.195x | 6.06 | 5.07 |

## Per-Level Histograms

- L1: compile_failed 21, correct 12, output_mismatch 10, runtime_error 10, shape_mismatch 1
- L2: compile_failed 30, correct 23, output_mismatch 31, runtime_error 6, shape_mismatch 5
- L3: compile_failed 30, correct 1, output_mismatch 10, runtime_error 6, shape_mismatch 1
- L4: compile_failed 3, runtime_error 6

## Notes

- Existing generated kernels only; no code generation was run.
- Eval logs are stored under `/tmp/kb_logs/` for this retime pass.
