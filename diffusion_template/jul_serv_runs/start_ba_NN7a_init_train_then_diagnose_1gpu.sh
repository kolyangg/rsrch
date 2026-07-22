#!/usr/bin/env bash
set -euo pipefail

# Train for 4k with 96-image validations, then run the deterministic 24-image matrix.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-ba_NN7a_init_1gpu}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"

RUN_FOREGROUND=1 \
RUN_NAME="${TRAIN_RUN_NAME}" \
NUM_EPOCHS=2 \
OPTIMIZER_STEPS_PER_EPOCH=2000 \
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
bash "${SCRIPT_DIR}/start_ba_NN7a_init_1gpu.sh" "$@"

CHECKPOINT="${PROJECT_DIR}/saved/${TRAIN_RUN_NAME}/checkpoint-epoch2.pth"
[[ -f "${CHECKPOINT}" ]] || {
  echo "Missing NN7a_init 4k checkpoint: ${CHECKPOINT}" >&2
  exit 3
}

CHECKPOINT_EPOCH=2 \
RUN_NAME="${TRAIN_RUN_NAME}_4000step_diagnostic24" \
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
BATCH_SIZE="${DIAGNOSTIC_BATCH_SIZE:-12}" \
SUBSET_SIZE="${DIAGNOSTIC_SUBSET_SIZE:-24}" \
SUBSET_SEED="${DIAGNOSTIC_SUBSET_SEED:-20260722}" \
bash "${SCRIPT_DIR}/start_ba_NN7a_init_checkpoint_reference_vs_noise_24_1gpu.sh" \
  "${CHECKPOINT}"
