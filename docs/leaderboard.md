# KernelBench CUDA Leaderboard

This leaderboard compares CUDA kernel generation runs on the same 206-problem
KernelBench dataset subset.

## Summary

| Rank | Model | Accuracy | Faster Than Reference | Top 3 Speedups |
| ---: | --- | ---: | ---: | --- |
| 1 | Codex GPT-5.5 | 142 / 206 = 68.93% | 42 / 142 = 29.58% | L2 P80 1089.219x; L2 P42 16.512x; L2 P83 15.145x |
| 2 | Deepseek V4 Pro | 94 / 206 = 45.63% | 50 / 89 = 56.18% | L2 P23 523.179x; L2 P18 27.248x; L1 P88 17.032x |
| 3 | Gemma4 26B | 38 / 206 = 18.45% | 23 / 38 = 60.53% | L2 P48 1.760x; L2 P87 1.710x; L2 P16 1.681x |
| 4 | Qwen3.6 35B A3B | 36 / 206 = 17.48% | 16 / 36 = 44.44% | L1 P88 8.900x; L2 P57 2.720x; L2 P7 1.909x |

## Source Runs

| Model | Run |
| --- | --- |
| Codex GPT-5.5 | `runs/ralph_codex_cuda_all` |
| Deepseek V4 Pro | `runs/openrouter_deepseek_v4_pro_leaderboard_l{1,2,3,4}` |
| Qwen3.6 35B A3B | `runs/ralph_qwen36_cuda_all_gpu1_server_isolated` |
| Gemma4 26B | `runs/ralph_gemma4_cuda_all` |

## Dataset

The original dataset had 270 problems. The leaderboard dataset records 206
problems.

| Set | Count |
| --- | ---: |
| Original dataset | 270 |
| Recorded dataset | 206 |

All scores below use the 206 recorded dataset problems as the denominator.

## Metrics

- `Accuracy`: correct generated kernels divided by `206`.
- `Compiled`: problems whose generated `ModelNew` reached a compiled/runnable
  eval state, as recorded in each score file.
- `Speedup`: `reference_runtime / generated_kernel_runtime`; values above
  `1.0x` are faster than the PyTorch reference.
- `FastP > 1.0 / 206`: number of correct kernels faster than the PyTorch
  reference divided by the full 206-problem denominator.
- `Reference`: the original PyTorch `Model` from each KernelBench problem file,
  not a separate handwritten CUDA kernel.

## Leaderboard

| Rank | Model | Correct | Accuracy | Evaluated | Compiled |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Codex GPT-5.5 | 142 / 206 | 68.93% | 206 | 190 |
| 2 | Deepseek V4 Pro | 94 / 206 | 45.63% | 206 | 145 |
| 3 | Gemma4 26B | 38 / 206 | 18.45% | 206 | 61 |
| 4 | Qwen3.6 35B A3B | 36 / 206 | 17.48% | 206 | 106 |

## Speed

Speed records come from retiming existing generated kernels. No code generation
was run during retime.

| Model | Correct Kernels With Speed Data | Geomean Speedup | Median Speedup | Faster Than Reference | FastP > 1.0 / 206 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Codex GPT-5.5 | 142 | 0.363x | 0.586x | 42 / 142 = 29.58% | 42 / 206 = 20.39% |
| Deepseek V4 Pro | 89 | 0.680x | 1.011x | 50 / 89 = 56.18% | 50 / 206 = 24.27% |
| Gemma4 26B | 38 | 0.642x | 1.005x | 23 / 38 = 60.53% | 23 / 206 = 11.17% |
| Qwen3.6 35B A3B | 36 | 0.437x | 0.946x | 16 / 36 = 44.44% | 16 / 206 = 7.77% |

### Speed By Level

| Model | L1 | L2 | L3 | L4 |
| --- | ---: | ---: | ---: | ---: |
| Gemma4 26B | 10 kernels, 0.305x geomean, 4 faster | 18 kernels, 0.917x geomean, 13 faster | 10 kernels, 0.713x geomean, 6 faster | 0 kernels |
| Qwen3.6 35B A3B | 12 kernels, 0.229x geomean, 2 faster | 23 kernels, 0.601x geomean, 14 faster | 1 kernel, 0.681x geomean, 0 faster | 0 kernels |

### Top Recorded Speedups

Top three correct kernels per model by recorded speedup.

| Model | Rank | Problem | Speedup | Reference Runtime | Kernel Runtime |
| --- | ---: | --- | ---: | ---: | ---: |
| Codex GPT-5.5 | 1 | L2 P80 | 1089.219x | 2.93 | 0.00269 |
| Codex GPT-5.5 | 2 | L2 P42 | 16.512x | 21.3 | 1.29 |
| Codex GPT-5.5 | 3 | L2 P83 | 15.145x | 7.3 | 0.482 |
| Deepseek V4 Pro | 1 | L2 P23 | 523.179x | 2.37 | 0.00453 |
| Deepseek V4 Pro | 2 | L2 P18 | 27.248x | 2.97 | 0.109 |
| Deepseek V4 Pro | 3 | L1 P88 | 17.032x | 10.1 | 0.593 |
| Gemma4 26B | 1 | L2 P48 | 1.760x | 6.6 | 3.75 |
| Gemma4 26B | 2 | L2 P87 | 1.710x | 21.2 | 12.4 |
| Gemma4 26B | 3 | L2 P16 | 1.681x | 34.8 | 20.7 |
| Qwen3.6 35B A3B | 1 | L1 P88 | 8.900x | 5.26 | 0.591 |
| Qwen3.6 35B A3B | 2 | L2 P57 | 2.720x | 8.05 | 2.96 |
| Qwen3.6 35B A3B | 3 | L2 P7 | 1.909x | 17.2 | 9.01 |

## Accuracy By Level

| Model | L1 | L2 | L3 | L4 |
| --- | ---: | ---: | ---: | ---: |
| Codex GPT-5.5 | 42 / 54 = 77.78% | 73 / 95 = 76.84% | 27 / 48 = 56.25% | 0 / 9 = 0.00% |
| Deepseek V4 Pro | 24 / 54 = 44.44% | 59 / 95 = 62.11% | 11 / 48 = 22.92% | 0 / 9 = 0.00% |
| Gemma4 26B | 10 / 54 = 18.52% | 18 / 95 = 18.95% | 10 / 48 = 20.83% | 0 / 9 = 0.00% |
| Qwen3.6 35B A3B | 12 / 54 = 22.22% | 23 / 95 = 24.21% | 1 / 48 = 2.08% | 0 / 9 = 0.00% |



## Error Type Histogram

Counts are restricted to the 206 recorded dataset problems.

For retime-only runs, missing kernel files are counted as `generation_error`.
Syntax, import, extension-build, and missing-`ModelNew` failures are counted as
`compile_failed`.

| Error Type | Codex GPT-5.5 | Deepseek V4 Pro | Qwen3.6 35B A3B | Gemma4 26B |
| --- | ---: | ---: | ---: | ---: |
| correct | 142 | 94 | 36 | 38 |
| output_mismatch | 46 | 32 | 51 | 19 |
| runtime_error | 15 | 24 | 28 | 4 |
| shape_mismatch | 1 | 7 | 7 | 3 |
| compile_failed | 2 | 21 | 84 | 131 |
| evaluation_error | 0 | 0 | 0 | 0 |
| static_failed | 0 | 7 | 0 | 0 |
| generation_error | 0 | 21 | 0 | 11 |
| extraction_failed | 0 | 0 | 0 | 0 |

Example Codex GPT-5.5 failures from this histogram:

| Error Type | Problem | Generated Kernel | Evidence |
| --- | --- | --- | --- |
| output_mismatch | L1 P50, `50_conv_standard_2D__square_input__square_kernel.py` | `runs/ralph_codex_cuda_all/generated_kernels/level_1/problem_50.py` | `fp32`, tolerance `atol=rtol=1e-4`; correctness trials `(0 / 5)`; max differences `0.000522`, `0.000545`, `0.000567`, `0.000548`, `0.000539`; avg differences all `0.000076`. |
| runtime_error | L1 P93, `93_masked_cumsum.py` | `runs/ralph_codex_cuda_all/generated_kernels/level_1/problem_93.py` | Raised `builtins.RuntimeError`: `expected scalar type Bool but found Float`. |

## Per-Level Histograms

### Codex GPT-5.5

| Level | Total | Histogram |
| --- | ---: | --- |
| L1 | 54 | correct 42, output_mismatch 9, runtime_error 3 |
| L2 | 95 | correct 73, output_mismatch 19, shape_mismatch 1, compile_failed 1, runtime_error 1 |
| L3 | 48 | correct 27, output_mismatch 18, compile_failed 1, runtime_error 2 |
| L4 | 9 | runtime_error 9 |

### Deepseek V4 Pro

| Level | Total | Histogram |
| --- | ---: | --- |
| L1 | 54 | correct 24, runtime_error 10, compile_failed 7, generation_error 7, output_mismatch 5, static_failed 1 |
| L2 | 95 | correct 59, output_mismatch 16, shape_mismatch 6, compile_failed 6, runtime_error 4, generation_error 4 |
| L3 | 48 | correct 11, output_mismatch 11, compile_failed 8, generation_error 6, runtime_error 6, static_failed 5, shape_mismatch 1 |
| L4 | 9 | runtime_error 4, generation_error 4, static_failed 1 |

### Qwen3.6 35B A3B

| Level | Total | Histogram |
| --- | ---: | --- |
| L1 | 54 | correct 12, output_mismatch 10, compile_failed 21, runtime_error 10, shape_mismatch 1 |
| L2 | 95 | correct 23, output_mismatch 31, compile_failed 30, shape_mismatch 5, runtime_error 6 |
| L3 | 48 | correct 1, output_mismatch 10, compile_failed 30, runtime_error 6, shape_mismatch 1 |
| L4 | 9 | compile_failed 3, runtime_error 6 |

### Gemma4 26B

| Level | Total | Histogram |
| --- | ---: | --- |
| L1 | 54 | correct 10, compile_failed 26, generation_error 6, output_mismatch 11, runtime_error 1 |
| L2 | 95 | compile_failed 66, shape_mismatch 3, correct 18, output_mismatch 4, generation_error 4 |
| L3 | 48 | compile_failed 32, correct 10, runtime_error 1, generation_error 1, output_mismatch 4 |
| L4 | 9 | compile_failed 7, runtime_error 2 |

## Recorded Dataset Problem IDs

These are the 206 problems used for the normalized leaderboard.

| Level | Count | Problem IDs |
| --- | ---: | --- |
| L1 | 54 | 1, 2, 3, 4, 6, 8, 10, 12, 13, 14, 15, 16, 17, 18, 40, 43, 47, 48, 49, 50, 51, 52, 53, 54, 56, 58, 60, 61, 62, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 77, 78, 79, 80, 81, 82, 83, 85, 86, 88, 93, 94, 95 |
| L2 | 95 | 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99 |
| L3 | 48 | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50 |
| L4 | 9 | 4, 5, 6, 7, 13, 14, 15, 16, 17 |

## Notes

- The detailed score records live under `docs/scores/{codex,deepseek_v4,qwen,gemma}/`.
- Codex, Qwen, and Gemma speed numbers are from retime passes over existing
  generated kernels. No new kernels were generated for those retime passes.
- Deepseek V4 Pro L4 generation used the local dataset because this workspace's
  HuggingFace KernelBench loader only supports levels 1, 2, and 3.
- For Deepseek V4 Pro, stale `Kernel not found` eval records are classified as
  generation or static-check failures rather than compile failures. This affects
  error subtype counts, not the `94 / 206` accuracy.
- `ref_runtime` is parsed from eval logs because `eval_results.json` does not
  persist reference timing for every run.
