#!/usr/bin/env bash
set -euo pipefail

# Five-way fixed-target NN4 checkpoint test:
# PM0, R1N1, R2N1, R1N2, R2N2. This is validation-only; the underlying
# resolved training schedule remains the standard 20k NN4 configuration.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-${1:-}}"
if [[ -z "${CHECKPOINT_PATH}" || ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Set CHECKPOINT_PATH or pass an NN4 checkpoint-epochN.pth as argument 1." >&2
  exit 2
fi
if [[ $# -gt 0 && "$1" == "${CHECKPOINT_PATH}" ]]; then
  shift
fi
CHECKPOINT_PATH="$(cd -- "$(dirname -- "${CHECKPOINT_PATH}")" && pwd)/$(basename -- "${CHECKPOINT_PATH}")"

CHECKPOINT_EPOCH="${CHECKPOINT_EPOCH:-}"
if [[ -z "${CHECKPOINT_EPOCH}" && "$(basename -- "${CHECKPOINT_PATH}")" =~ checkpoint-epoch([0-9]+)\.pth$ ]]; then
  CHECKPOINT_EPOCH="${BASH_REMATCH[1]}"
fi
if [[ ! "${CHECKPOINT_EPOCH}" =~ ^[0-9]+$ ]]; then
  echo "Could not infer checkpoint epoch; set CHECKPOINT_EPOCH explicitly." >&2
  exit 2
fi

export RUN_FOREGROUND="${RUN_FOREGROUND:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MASTER_PORT="${MASTER_PORT:-29643}"
export NN4_VALIDATION_MODEL="${NN4_VALIDATION_MODEL:-SG161222/RealVisXL_V4.0}"

CHECKPOINT_STEP=$((CHECKPOINT_EPOCH * 2000))
if [[ "${NN4_VALIDATION_MODEL}" == "null" ]]; then
  VALIDATION_LABEL="same_sdxl"
else
  VALIDATION_LABEL="realvis"
fi
export RUN_NAME="${RUN_NAME:-ba_NN4_${CHECKPOINT_STEP}step_${VALIDATION_LABEL}_reference_vs_noise}"

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/ppr_NN4_${CHECKPOINT_STEP}step_${VALIDATION_LABEL}_reference_vs_noise}"
BATCH_SIZE="${BATCH_SIZE:-12}"
LIMIT="${LIMIT:-96}"
NOISE_SEEDS="${NOISE_SEEDS:-[918273,271828]}"

exec bash "${SCRIPT_DIR}/start_ba_NN4_causal_null_up0_realvis_1gpu.sh" \
  validation_only=true \
  continue_run=false \
  saved_checkpoint="${CHECKPOINT_PATH}" \
  ppr_checkpoint_require_nonzero=true \
  strict_checkpoint_model_config=true \
  ppr_expected_checkpoint_epoch="${CHECKPOINT_EPOCH}" \
  ppr_reference_noise_test=true \
  ppr_reference_noise_output_dir="${OUTPUT_DIR}" \
  ppr_reference_noise_overwrite="${OVERWRITE_OUTPUT:-false}" \
  ppr_reference_noise_seeds="${NOISE_SEEDS}" \
  ppr_scale_sweep=false \
  ppr_diagnostic_matrix=false \
  datasets.val.manual_val.limit="${LIMIT}" \
  dataloaders.manual_val.batch_size="${BATCH_SIZE}" \
  "$@"
