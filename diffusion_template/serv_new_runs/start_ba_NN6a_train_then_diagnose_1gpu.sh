#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-ba_NN6a_factorized_identity_only_up0_nfs_1gpu}"
export NN6_RUN_NAME="${TRAIN_RUN_NAME}"
export NUM_EPOCHS=2
export STEPS_PER_EPOCH=2000
bash "${SCRIPT_DIR}/start_ba_NN6a_factorized_identity_only_up0_1gpu.sh"

for CHECKPOINT_EPOCH in 1 2; do
    CHECKPOINT_STEP=$((CHECKPOINT_EPOCH * 2000))
    CHECKPOINT="${PROJECT_DIR}/saved/${TRAIN_RUN_NAME}/checkpoint-epoch${CHECKPOINT_EPOCH}.pth"
    [[ -f "${CHECKPOINT}" ]] || {
        echo "Missing NN6a ${CHECKPOINT_STEP}-step checkpoint: ${CHECKPOINT}" >&2
        exit 3
    }
    CHECKPOINT_EPOCH="${CHECKPOINT_EPOCH}" \
    RUN_NAME="${TRAIN_RUN_NAME}_${CHECKPOINT_STEP}step_diagnostic" \
    OUTPUT_DIR="${PROJECT_DIR}/ppr_${TRAIN_RUN_NAME}_${CHECKPOINT_STEP}step_realvis_scale1_reference_vs_noise" \
    CUDA_VISIBLE_DEVICES=0 \
    BATCH_SIZE="${DIAGNOSTIC_BATCH_SIZE:-12}" \
    bash "${SCRIPT_DIR}/start_ba_NN6a_checkpoint_reference_vs_noise_1gpu.sh" \
        "${CHECKPOINT}"
done
