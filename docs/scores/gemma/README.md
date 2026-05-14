# Gemma4 26B KernelBench Retime Score

Existing generated kernels retimed on the 206-problem leaderboard dataset. No code generation was run.

## Overall Score

```text
correct: 38 / 206
accuracy: 18.45%
evaluated: 206 / 206
compiled: 61 / 206
compile_rate: 29.61%
```

Status counts:

```text
compile_failed: 131
correct: 38
generation_error: 11
output_mismatch: 19
runtime_error: 4
shape_mismatch: 3
```

## By Level

| Level | Total | Evaluated | Compiled | Correct | Accuracy | Compile Rate |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 54 | 54 | 22 | 10 | 18.52% | 40.74% |
| 2 | 95 | 95 | 25 | 18 | 18.95% | 26.32% |
| 3 | 48 | 48 | 14 | 10 | 20.83% | 29.17% |
| 4 | 9 | 9 | 0 | 0 | 0.00% | 0.00% |

## Speed

Speedup is `reference_runtime / generated_kernel_runtime`, parsed from retime eval logs.

| Scope | Kernels | Geomean Speedup | Median Speedup | Faster Than Reference |
|---|---:|---:|---:|---:|
| All | 38 | 0.642x | 1.005x | 23 / 38 |
| L1 | 10 | 0.305x | 0.459x | 4 / 10 |
| L2 | 18 | 0.917x | 1.084x | 13 / 18 |
| L3 | 10 | 0.713x | 1.012x | 6 / 10 |
| L4 | 0 | 0.000x | 0.000x | 0 / 0 |

| Threshold | Count | Score |
|---:|---:|---:|
| >1.0x | 23 / 206 | 11.17% |
| >1.5x | 3 / 206 | 1.46% |
| >2.0x | 0 / 206 | 0.00% |

## Top Speedups

| Problem | Speedup | Ref Runtime | Kernel Runtime |
|---|---:|---:|---:|
| L2 P48 | 1.760x | 6.6 | 3.75 |
| L2 P87 | 1.710x | 21.2 | 12.4 |
| L2 P16 | 1.681x | 34.8 | 20.7 |
| L2 P69 | 1.382x | 4.09 | 2.96 |
| L2 P58 | 1.351x | 27.3 | 20.2 |
| L2 P37 | 1.316x | 11.9 | 9.04 |
| L2 P35 | 1.250x | 14.5 | 11.6 |
| L1 P43 | 1.195x | 6.06 | 5.07 |
| L2 P60 | 1.123x | 30.1 | 26.8 |
| L2 P96 | 1.107x | 22.8 | 20.6 |

## Per-Level Histograms

- L1: compile_failed 26, correct 10, generation_error 6, output_mismatch 11, runtime_error 1
- L2: compile_failed 66, correct 18, generation_error 4, output_mismatch 4, shape_mismatch 3
- L3: compile_failed 32, correct 10, generation_error 1, output_mismatch 4, runtime_error 1
- L4: compile_failed 7, runtime_error 2

## Notes

- Existing generated kernels only; no code generation was run.
- Eval logs are stored under `/tmp/kb_logs/` for this retime pass.
