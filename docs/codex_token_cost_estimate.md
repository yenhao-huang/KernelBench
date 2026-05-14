# Codex Token Cost Estimate

This document records token-count estimates for the saved Codex KernelBench
runs. These are tokenizer estimates from saved prompts and raw generations, not
provider billing records.

## Method

Input tokens were estimated by reconstructing the Codex CLI prompt:

```text
Codex hard-requirements wrapper + KernelBench backend prompt + problem source
```

Output tokens were estimated from:

```text
runs/<run_name>/raw_generation/level_*/problem_*.txt
```

Extracted-code tokens were estimated from:

```text
runs/<run_name>/generated_kernels/level_*/problem_*.py
```

Counts use LiteLLM token counting with `model=gpt-5.5` when available. Treat
them as planning estimates for `input_token` and `output_token`.

## Summary

| Run | Problems | Input Tokens | Output Tokens | Total Tokens | Avg Input / Problem | Avg Output / Problem |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `runs/ralph_codex_cuda_10` | 3 | 3,182 | 2,026 | 5,208 | 1,060.7 | 675.3 |
| `runs/ralph_codex_cuda_10_v2` | 10 | 10,597 | 8,825 | 19,422 | 1,059.7 | 882.5 |
| `runs/ralph_codex_cuda_all` | 270 | 336,973 | 309,167 | 646,140 | 1,248.0 | 1,145.1 |

## Distribution

| Run | Input Median | Input P90 | Input Min | Input Max | Output Median | Output P90 | Output Min | Output Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `runs/ralph_codex_cuda_10` | 1,051 | 1,084 | 1,047 | 1,084 | 695 | 708 | 623 | 708 |
| `runs/ralph_codex_cuda_10_v2` | 1,053 | 1,084 | 1,038 | 1,095 | 818.5 | 1,131 | 626 | 1,278 |
| `runs/ralph_codex_cuda_all` | 1,137.5 | 1,561 | 989 | 6,744 | 1,051.5 | 1,797 | 275 | 3,629 |

## Rule Of Thumb

For this KernelBench Codex setup:

```text
input_token  ~= 1,250 / problem on average
output_token ~= 1,150 / problem on average
total_token  ~= 2,400 / problem on average
```

For budget planning, a conservative per-problem estimate is:

```text
input_token  = 1,600
output_token = 1,800
total_token  = 3,400
```

For a full 270-problem run:

```text
average estimate      ~= 646k total tokens
conservative estimate ~= 918k total tokens
```

## Cost Formula

Use the model provider's current pricing:

```text
cost =
  input_token  / 1_000_000 * input_price_per_1m_tokens
+ output_token / 1_000_000 * output_price_per_1m_tokens
```

Example for the saved full Codex run:

```text
cost =
  336,973 / 1_000_000 * input_price_per_1m_tokens
+ 309,167 / 1_000_000 * output_price_per_1m_tokens
```

## Notes

- These estimates do not include hidden reasoning tokens, retry overhead, or
  provider-side accounting differences.
- The saved Codex runs did not record exact provider usage metadata.
- Output tokens are based on raw generation files. Extracted kernel totals are
  slightly lower because markdown fences and any wrapper text are removed.
- The 270-problem run has larger prompts for some level 3 and level 4 problems;
  that is why `input_max` is much higher than the median.
