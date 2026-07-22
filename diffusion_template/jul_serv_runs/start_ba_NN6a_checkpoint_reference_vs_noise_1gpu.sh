#!/usr/bin/env bash
set -euo pipefail

# Primary-server RealVis scale-1 causal matrix with exact identity-noise checks.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${1:-}}"
if [[ -z "${CHECKPOINT_PATH}" || ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Set CHECKPOINT_PATH or pass checkpoint-epoch1/2.pth as argument 1." >&2
  exit 2
fi
if [[ $# -gt 0 && "$1" == "${CHECKPOINT_PATH}" ]]; then shift; fi
CHECKPOINT_PATH="$(cd -- "$(dirname -- "${CHECKPOINT_PATH}")" && pwd)/$(basename -- "${CHECKPOINT_PATH}")"

CHECKPOINT_EPOCH="${CHECKPOINT_EPOCH:-}"
if [[ -z "${CHECKPOINT_EPOCH}" && "$(basename -- "${CHECKPOINT_PATH}")" =~ checkpoint-epoch([0-9]+)\.pth$ ]]; then
  CHECKPOINT_EPOCH="${BASH_REMATCH[1]}"
fi
[[ "${CHECKPOINT_EPOCH}" =~ ^[12]$ ]] || {
  echo "NN6a approval expects checkpoint epoch 1 (2k) or 2 (4k)." >&2
  exit 2
}

CHECKPOINT_STEP=$((CHECKPOINT_EPOCH * 2000))
export RUN_FOREGROUND=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MASTER_PORT="${MASTER_PORT:-29663}"
export NUM_EPOCHS="${CHECKPOINT_EPOCH}"
export RUN_NAME="${RUN_NAME:-ba_NN6a_${CHECKPOINT_STEP}step_realvis_scale1_reference_vs_noise}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/ppr_NN6a_${CHECKPOINT_STEP}step_realvis_scale1_reference_vs_noise}"

exec bash "${SCRIPT_DIR}/start_ba_NN6a_factorized_identity_only_up0_1gpu.sh" \
  validation_only=true \
  continue_run=false \
  saved_checkpoint="${CHECKPOINT_PATH}" \
  ppr_checkpoint_require_nonzero=true \
  strict_checkpoint_model_config=true \
  ppr_expected_checkpoint_epoch="${CHECKPOINT_EPOCH}" \
  ppr_reference_noise_test=true \
  ppr_reference_noise_scale=1.0 \
  ppr_identity_noise_tolerance="${IDENTITY_NOISE_TOLERANCE:-0.0}" \
  ppr_reference_noise_output_dir="${OUTPUT_DIR}" \
  ppr_reference_noise_overwrite="${OVERWRITE_OUTPUT:-false}" \
  ppr_reference_noise_seeds="${NOISE_SEEDS:-[918273,271828]}" \
  datasets.val.manual_val.limit="${LIMIT:-96}" \
  dataloaders.manual_val.batch_size="${BATCH_SIZE:-12}" \
  "$@"
