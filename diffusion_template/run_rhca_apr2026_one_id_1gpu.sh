#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export ACCELERATE_LOG_LEVEL="${ACCELERATE_LOG_LEVEL:-error}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export DIFFUSERS_VERBOSITY="${DIFFUSERS_VERBOSITY:-error}"
export COMET_DISABLE_AUTO_LOGGING="${COMET_DISABLE_AUTO_LOGGING:-1}"
export COMET_LOGGING_CONSOLE="${COMET_LOGGING_CONSOLE:-ERROR}"

WRITER="${WRITER:-cometml}"
RUN_NAME="${RUN_NAME:-rhca_1e-4_ml_step2_allst_trref_diff_replay}"
PM_PATH="${PM_PATH:-}"

if [[ "${WRITER}" == "cometml" && -z "${COMET_API_KEY:-}" ]]; then
  echo "COMET_API_KEY must be exported when WRITER=cometml." >&2
  echo "For an offline smoke test, run with WRITER=console." >&2
  exit 2
fi

MODEL_OVERRIDES=()
if [[ -n "${PM_PATH}" ]]; then
  MODEL_OVERRIDES+=("model.photomaker_path=${PM_PATH}")
fi

accelerate launch \
  --config_file=src/configs/ddp/accelerate.yaml \
  --num_processes=1 \
  train.py \
  --config-name=one_id_rhca_apr2026_replay \
  "writer=${WRITER}" \
  "writer.run_name=${RUN_NAME}" \
  "${MODEL_OVERRIDES[@]}" \
  "$@"
