#!/usr/bin/env bash
set -euo pipefail

cd /workspace/external/KernelBench

MODEL_NAME="${MODEL_NAME:-openrouter/deepseek/deepseek-v4-pro}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
TEMPERATURE="${TEMPERATURE:-0.0}"
RUN_PREFIX="${RUN_PREFIX:-openrouter_deepseek_v4_pro_leaderboard}"
CUDA_ENV_PATH="/workspace/.venv/lib/python3.11/site-packages/nvidia/cu13"

export PATH="/workspace/external/KernelBench/.venv/bin:${CUDA_ENV_PATH}/bin:${PATH}"
export CUDA_HOME="${CUDA_ENV_PATH}"
export LD_LIBRARY_PATH="${CUDA_ENV_PATH}/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

mkdir -p "runs/${RUN_PREFIX}_logs"
LOG_PATH="runs/${RUN_PREFIX}_logs/leaderboard_run.log"

run_level() {
  local level="$1"
  local ids="$2"
  local workers="${3:-$NUM_WORKERS}"
  local run_name="${RUN_PREFIX}_l${level}"
  local generation_dataset_src="huggingface"
  if [[ "${level}" == "4" ]]; then
    generation_dataset_src="local"
  fi

  echo "================================================================"
  echo "[LEVEL ${level}] run_name=${run_name}"
  echo "[LEVEL ${level}] problem_ids=${ids}"
  echo "[LEVEL ${level}] generation_dataset_src=${generation_dataset_src}"
  echo "================================================================"

  uv run python scripts/generate_samples.py \
    run_name="${run_name}" \
    dataset_src="${generation_dataset_src}" \
    level="${level}" \
    server_type=openrouter \
    model_name="${MODEL_NAME}" \
    problem_ids="${ids}" \
    num_workers="${workers}" \
    temperature="${TEMPERATURE}" \
    max_tokens="${MAX_TOKENS}"

  uv run python scripts/eval_from_generations.py \
    run_name="${run_name}" \
    dataset_src=local \
    level="${level}" \
    problem_ids="${ids}" \
    num_gpu_devices=1 \
    timeout=300 \
    measure_performance=True
}

# Reuse the existing five-problem smoke test artifacts if present, so restarting
# this full run does not spend OpenRouter credits on those problems again.
if [[ -d runs/openrouter_deepseek_v4_pro_5 ]]; then
  mkdir -p "runs/${RUN_PREFIX}_l1"
  cp -n runs/openrouter_deepseek_v4_pro_5/level_1_problem_*_sample_0_kernel.py "runs/${RUN_PREFIX}_l1/" 2>/dev/null || true
  if [[ -f runs/openrouter_deepseek_v4_pro_5/eval_results.json && ! -f "runs/${RUN_PREFIX}_l1/eval_results.json" ]]; then
    cp runs/openrouter_deepseek_v4_pro_5/eval_results.json "runs/${RUN_PREFIX}_l1/eval_results.json"
  fi
fi

{
  echo "[START] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[MODEL] ${MODEL_NAME}"
  echo "[CUDA_VISIBLE_DEVICES] ${CUDA_VISIBLE_DEVICES}"

  run_level 1 "1,2,3,4,6,8,10,12,13,14,15,16,17,18,40,43,47,48,49,50,51,52,53,54,56,58,60,61,62,64,65,66,67,68,69,70,71,72,73,74,75,77,78,79,80,81,82,83,85,86,88,93,94,95" "${NUM_WORKERS}"
  run_level 2 "5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99" "${NUM_WORKERS}"
  run_level 3 "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,19,20,21,22,23,24,25,26,27,28,29,30,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50" "${NUM_WORKERS}"
  run_level 4 "4,5,6,7,13,14,15,16,17" 2

  echo "[DONE] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee -a "${LOG_PATH}"
