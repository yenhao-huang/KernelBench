#!/usr/bin/env python3
"""
Run a resumable full KernelBench CUDA generation/evaluation pass for Gemma4.

This script is intended to be launched by /workspace/core/cli/ralph_loop.py.
It measures accuracy, so it runs correctness checks but not performance timing.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.eval import eval_kernel_against_ref, get_torch_dtype_from_string
from kernelbench.kernel_static_checker import validate_kernel_static
from kernelbench.prompt_constructor_toml import get_custom_prompt, get_prompt_for_backend
from kernelbench.utils import (
    create_inference_server_from_presets,
    extract_first_code,
    set_gpu_arch,
)


REPO_TOP_DIR = Path(__file__).resolve().parents[1]


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if hasattr(value, "__dict__"):
        return str(value)
    return str(value)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(data, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, sort_keys=True, default=_json_default) + "\n")


def _normalize_optional(value: Any) -> Any:
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return value


def _build_prompt(args: argparse.Namespace, ref_arch_src: str) -> str:
    custom_prompt_key = _normalize_optional(args.custom_prompt_key)
    include_hardware = args.include_hardware_info
    if custom_prompt_key:
        return get_custom_prompt(
            custom_prompt_key,
            ref_arch_src=ref_arch_src,
            backend=args.backend,
            option=args.prompt_option,
            precision=args.precision,
            include_hardware=include_hardware,
            gpu_name=_normalize_optional(args.hardware_gpu_name),
        )
    return get_prompt_for_backend(
        ref_arch_src,
        args.backend,
        option=args.prompt_option,
        precision=args.precision,
        include_hardware=include_hardware,
        gpu_name=_normalize_optional(args.hardware_gpu_name),
    )


def _query_codex_cli(prompt: str, args: argparse.Namespace) -> str:
    codex_prompt = (
        "You are generating a KernelBench CUDA solution. Return only one fenced "
        "python code block defining ModelNew. Do not edit files. Do not run "
        "commands. Do not include explanations.\n\n"
        "Hard requirements:\n"
        "- Write at least one explicit CUDA __global__ kernel in cuda_sources.\n"
        "- Use torch.utils.cpp_extension.load_inline to compile it.\n"
        "- Do not call torch.matmul, torch.mm, torch.bmm, torch.einsum, or "
        "torch.nn.functional compute ops in ModelNew.forward.\n"
        "- Do not use cuBLAS, CUTLASS, Thrust, or other library matmul helpers.\n"
        "- Do not add try/except fallback paths.\n"
        "- Assume inputs are CUDA tensors with the dtype requested by the prompt.\n\n"
        f"{prompt}"
    )
    command = [
        *args.codex_command.split(),
        "exec",
        "--sandbox",
        "read-only",
        "--skip-git-repo-check",
        "--cd",
        str(REPO_TOP_DIR),
    ]
    if args.codex_model:
        command.extend(["--model", args.codex_model])

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "last_message.txt"
        command.extend(["--output-last-message", str(output_path), "-"])
        started_at = time.time()
        completed = subprocess.run(
            command,
            input=codex_prompt,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=args.codex_timeout_sec,
            cwd=str(REPO_TOP_DIR),
        )
        elapsed = time.time() - started_at
        print(f"[Timing] Codex inference took {elapsed:.2f} seconds", flush=True)
        if completed.returncode != 0:
            raise RuntimeError(
                "Codex CLI generation failed with exit code "
                f"{completed.returncode}: {(completed.stdout or '')[-4000:]}"
            )
        if output_path.exists():
            output = output_path.read_text(encoding="utf-8", errors="ignore")
            if output.strip():
                return output
        return completed.stdout or ""


def _create_inference_server(args: argparse.Namespace):
    if args.server_type == "codex_cli":
        return lambda prompt: _query_codex_cli(prompt, args)
    return create_inference_server_from_presets(
        server_type=args.server_type,
        model_name=args.model_name,
        server_address=args.server_address,
        server_port=args.server_port,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        verbose=args.verbose,
        time_generation=True,
    )


def _problem_result_path(results_dir: Path, level: int, problem_id: int) -> Path:
    return results_dir / "per_problem" / f"level_{level}" / f"problem_{problem_id}.json"


def _result_already_done(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return data.get("status") in {
        "evaluated",
        "static_failed",
        "extraction_failed",
        "generation_error",
        "evaluation_error",
    }


def _summarize(results: list[dict[str, Any]], expected_total: int) -> dict[str, Any]:
    completed = len(results)
    evaluated = [r for r in results if r.get("status") == "evaluated"]
    compiled = [
        r for r in evaluated if (r.get("eval_result") or {}).get("compiled") is True
    ]
    correct = [
        r for r in evaluated if (r.get("eval_result") or {}).get("correctness") is True
    ]
    status_counts: dict[str, int] = {}
    for result in results:
        status = str(result.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

    return {
        "expected_total": expected_total,
        "completed": completed,
        "is_complete": completed == expected_total,
        "evaluated": len(evaluated),
        "compiled": len(compiled),
        "correct": len(correct),
        "accuracy_all": len(correct) / expected_total if expected_total else 0.0,
        "accuracy_evaluated": len(correct) / len(evaluated) if evaluated else 0.0,
        "compile_rate_all": len(compiled) / expected_total if expected_total else 0.0,
        "status_counts": status_counts,
    }


def _load_all_results(results_dir: Path) -> list[dict[str, Any]]:
    loaded: list[dict[str, Any]] = []
    for path in sorted((results_dir / "per_problem").glob("level_*/problem_*.json")):
        try:
            loaded.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return loaded


def _eval_kernel_isolated(
    *,
    ref_arch_src: str,
    custom_kernel: str,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        ref_path = tmpdir_path / "ref.py"
        kernel_path = tmpdir_path / "kernel.py"
        output_path = tmpdir_path / "eval_result.json"
        ref_path.write_text(ref_arch_src, encoding="utf-8")
        kernel_path.write_text(custom_kernel, encoding="utf-8")

        child_code = r"""
import json
import sys
import traceback
from pathlib import Path

from kernelbench.eval import eval_kernel_against_ref, get_torch_dtype_from_string

ref_path = Path(sys.argv[1])
kernel_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])
backend = sys.argv[4]
precision = sys.argv[5]
num_correct_trials = int(sys.argv[6])
verbose = sys.argv[7] == "1"

try:
    result = eval_kernel_against_ref(
        ref_path.read_text(encoding="utf-8"),
        kernel_path.read_text(encoding="utf-8"),
        verbose=verbose,
        measure_performance=False,
        num_correct_trials=num_correct_trials,
        backend=backend,
        precision=get_torch_dtype_from_string(precision),
    )
    payload = None if result is None else result.model_dump()
    output_path.write_text(json.dumps({"ok": True, "result": payload}), encoding="utf-8")
except Exception as exc:
    output_path.write_text(
        json.dumps(
            {
                "ok": False,
                "error_type": f"{exc.__class__.__module__}.{exc.__class__.__name__}",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        ),
        encoding="utf-8",
    )
"""
        command = [
            sys.executable,
            "-c",
            child_code,
            str(ref_path),
            str(kernel_path),
            str(output_path),
            args.backend,
            args.precision,
            str(args.num_correct_trials),
            "1" if args.verbose else "0",
        ]
        completed = subprocess.run(
            command,
            cwd=str(REPO_TOP_DIR),
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.eval_timeout_sec,
        )
        if not output_path.exists():
            return {
                "ok": False,
                "error_type": "subprocess.NoResult",
                "error": (
                    f"isolated evaluator exited {completed.returncode} without "
                    f"writing a result. Output: {(completed.stdout or '')[-4000:]}"
                ),
            }
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        if not payload.get("ok"):
            payload["subprocess_returncode"] = completed.returncode
            payload["subprocess_output_tail"] = (completed.stdout or "")[-4000:]
        return payload


def _evaluate_one(
    *,
    args: argparse.Namespace,
    inference_server,
    results_dir: Path,
    level: int,
    problem,
) -> dict[str, Any]:
    problem_id = int(problem.problem_id)
    started_at = time.time()
    raw_dir = results_dir / "raw_generation" / f"level_{level}"
    kernel_dir = results_dir / "generated_kernels" / f"level_{level}"
    raw_path = raw_dir / f"problem_{problem_id}.txt"
    kernel_path = kernel_dir / f"problem_{problem_id}.py"

    base: dict[str, Any] = {
        "level": level,
        "problem_id": problem_id,
        "problem_name": problem.name,
        "backend": args.backend,
        "precision": args.precision,
        "prompt_option": args.prompt_option,
        "model_name": args.model_name,
        "raw_generation_path": str(raw_path.relative_to(REPO_TOP_DIR)),
        "generated_kernel_path": str(kernel_path.relative_to(REPO_TOP_DIR)),
    }

    try:
        prompt = _build_prompt(args, problem.code)
        raw_generation = inference_server(prompt)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(raw_generation or "", encoding="utf-8")

        custom_kernel = extract_first_code(raw_generation, ["python", "cpp"])
        if custom_kernel is None:
            result = {
                **base,
                "status": "extraction_failed",
                "elapsed_sec": round(time.time() - started_at, 3),
                "error": "No extractable code block or raw Python/CUDA module found.",
            }
            _write_json(_problem_result_path(results_dir, level, problem_id), result)
            return result

        kernel_path.parent.mkdir(parents=True, exist_ok=True)
        kernel_path.write_text(custom_kernel, encoding="utf-8")

        static_ok, static_errors, static_warnings = validate_kernel_static(
            custom_kernel,
            backend=args.backend,
            precision=args.precision,
        )
        if not static_ok:
            result = {
                **base,
                "status": "static_failed",
                "elapsed_sec": round(time.time() - started_at, 3),
                "static_errors": static_errors,
                "static_warnings": static_warnings,
            }
            _write_json(_problem_result_path(results_dir, level, problem_id), result)
            return result

        if args.isolate_eval:
            isolated_result = _eval_kernel_isolated(
                ref_arch_src=problem.code,
                custom_kernel=custom_kernel,
                args=args,
            )
            if isolated_result is None or not isolated_result.get("ok"):
                result = {
                    **base,
                    "status": "evaluation_error",
                    "elapsed_sec": round(time.time() - started_at, 3),
                    "error": "Isolated KernelBench evaluator failed.",
                    "static_warnings": static_warnings,
                    "isolated_eval": isolated_result,
                }
                _write_json(_problem_result_path(results_dir, level, problem_id), result)
                return result
            eval_result = isolated_result.get("result")
        else:
            eval_result = eval_kernel_against_ref(
                problem.code,
                custom_kernel,
                verbose=args.verbose,
                measure_performance=False,
                num_correct_trials=args.num_correct_trials,
                backend=args.backend,
                precision=get_torch_dtype_from_string(args.precision),
            )
        if eval_result is None:
            result = {
                **base,
                "status": "evaluation_error",
                "elapsed_sec": round(time.time() - started_at, 3),
                "error": "KernelBench evaluator returned None.",
                "static_warnings": static_warnings,
            }
            _write_json(_problem_result_path(results_dir, level, problem_id), result)
            return result

        result = {
            **base,
            "status": "evaluated",
            "elapsed_sec": round(time.time() - started_at, 3),
            "static_warnings": static_warnings,
            "eval_result": eval_result if isinstance(eval_result, dict) else eval_result.model_dump(),
        }
        _write_json(_problem_result_path(results_dir, level, problem_id), result)
        return result
    except Exception as exc:
        result = {
            **base,
            "status": "generation_error",
            "elapsed_sec": round(time.time() - started_at, 3),
            "error_type": f"{exc.__class__.__module__}.{exc.__class__.__name__}",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(_problem_result_path(results_dir, level, problem_id), result)
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/ralph_gemma4_cuda_all")
    parser.add_argument("--levels", default="1,2,3,4")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dataset-src", default="local")
    parser.add_argument("--dataset-name", default="ScalingIntelligence/KernelBench")
    parser.add_argument("--server-type", default="local")
    parser.add_argument("--model-name", default="gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf")
    parser.add_argument("--server-address", default="192.168.1.78")
    parser.add_argument("--server-port", type=int, default=3132)
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--gpu-arch", default="Ada")
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--codex-command", default="codex")
    parser.add_argument("--codex-model", default=None)
    parser.add_argument("--codex-timeout-sec", type=int, default=900)
    parser.add_argument("--prompt-option", default="one_shot")
    parser.add_argument("--custom-prompt-key", default=None)
    parser.add_argument("--hardware-gpu-name", default=None)
    parser.add_argument("--include-hardware-info", action="store_true")
    parser.add_argument("--num-correct-trials", type=int, default=5)
    parser.add_argument("--isolate-eval", action="store_true")
    parser.add_argument("--eval-timeout-sec", type=int, default=900)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.backend = args.backend.lower()
    args.prompt_option = args.prompt_option.lower()
    results_dir = (REPO_TOP_DIR / args.output_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    levels = [int(part.strip()) for part in args.levels.split(",") if part.strip()]
    set_gpu_arch([args.gpu_arch])

    inference_server = _create_inference_server(args)

    datasets = {
        level: construct_kernelbench_dataset(
            level=level,
            source=args.dataset_src,
            dataset_name=args.dataset_name,
        )
        for level in levels
    }
    selected_problems = []
    for level in levels:
        for problem in datasets[level]:
            selected_problems.append((level, problem))
    if args.limit is not None:
        selected_problems = selected_problems[: args.limit]
    expected_total = len(selected_problems)

    config_snapshot = {
        "levels": levels,
        "limit": args.limit,
        "expected_total": expected_total,
        "dataset_src": args.dataset_src,
        "server_type": args.server_type,
        "model_name": args.model_name,
        "server_address": args.server_address,
        "server_port": args.server_port,
        "backend": args.backend,
        "gpu_arch": args.gpu_arch,
        "precision": args.precision,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "codex_command": args.codex_command,
        "codex_model": args.codex_model,
        "codex_timeout_sec": args.codex_timeout_sec,
        "prompt_option": args.prompt_option,
        "num_correct_trials": args.num_correct_trials,
        "measure_performance": False,
    }
    _write_json(results_dir / "config_snapshot.json", config_snapshot)

    jsonl_path = results_dir / "results.jsonl"
    for level, problem in selected_problems:
            result_path = _problem_result_path(results_dir, level, int(problem.problem_id))
            if args.resume and not args.force and _result_already_done(result_path):
                print(f"[SKIP] level={level} problem_id={problem.problem_id}", flush=True)
                continue

            print(
                f"[RUN] level={level} problem_id={problem.problem_id} name={problem.name}",
                flush=True,
            )
            result = _evaluate_one(
                args=args,
                inference_server=inference_server,
                results_dir=results_dir,
                level=level,
                problem=problem,
            )
            _append_jsonl(jsonl_path, result)
            status = result.get("status")
            eval_result = result.get("eval_result") or {}
            print(
                "[RESULT] "
                f"level={level} problem_id={problem.problem_id} status={status} "
                f"compiled={eval_result.get('compiled')} "
                f"correctness={eval_result.get('correctness')}",
                flush=True,
            )

            summary = _summarize(_load_all_results(results_dir), expected_total)
            _write_json(results_dir / "summary.json", summary)
            print(
                "[SUMMARY] "
                f"completed={summary['completed']}/{summary['expected_total']} "
                f"correct={summary['correct']} "
                f"accuracy_all={summary['accuracy_all']:.4f}",
                flush=True,
            )

    summary = _summarize(_load_all_results(results_dir), expected_total)
    _write_json(results_dir / "summary.json", summary)
    if not summary["is_complete"]:
        print(f"Incomplete run: {summary}", flush=True)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
