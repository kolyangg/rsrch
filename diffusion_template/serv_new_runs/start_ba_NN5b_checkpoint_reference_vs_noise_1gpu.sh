#!/usr/bin/env bash
set -euo pipefail

# Same-SDXL, scale-1 five-way approval test for an NN5b 2k/4k checkpoint.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-${1:-}}"
if [[ -z "${CHECKPOINT_PATH}" || ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "Set CHECKPOINT_PATH or pass checkpoint-epoch1/2.pth as argument 1." >&2
    exit 2
fi
if [[ $# -gt 0 && "$1" == "${CHECKPOINT_PATH}" ]]; then
    shift
fi
CHECKPOINT_PATH="$(
    cd -- "$(dirname -- "${CHECKPOINT_PATH}")"
    pwd
)/$(basename -- "${CHECKPOINT_PATH}")"

CHECKPOINT_EPOCH="${CHECKPOINT_EPOCH:-}"
if [[ -z "${CHECKPOINT_EPOCH}" \
      && "$(basename -- "${CHECKPOINT_PATH}")" =~ checkpoint-epoch([0-9]+)\.pth$ ]]; then
    CHECKPOINT_EPOCH="${BASH_REMATCH[1]}"
fi
[[ "${CHECKPOINT_EPOCH}" =~ ^[12]$ ]] || {
    echo "NN5b approval expects epoch 1 (2k) or epoch 2 (4k)." >&2
    exit 2
}

CHECKPOINT_STEP=$((CHECKPOINT_EPOCH * 2000))
export NN5_NUM_PROCESSES=1
export NN5_RUN_NAME="${RUN_NAME:-ba_NN5b_${CHECKPOINT_STEP}step_same_sdxl_scale1_reference_vs_noise}"
export NN5_CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NN5_MASTER_PORT="${MASTER_PORT:-29664}"
export GLOBAL_EFFECTIVE_BATCH=2
export TRAIN_BATCH_SIZE=1
export STEPS_PER_EPOCH=2000
export NUM_EPOCHS=2

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/ppr_NN5b_${CHECKPOINT_STEP}step_same_sdxl_scale1_reference_vs_noise}"

exec bash "${SCRIPT_DIR}/_start_ba_NN5b_server_common.sh" \
    validation_only=true \
    continue_run=false \
    saved_checkpoint="${CHECKPOINT_PATH}" \
    ppr_checkpoint_require_nonzero=true \
    strict_checkpoint_model_config=true \
    ppr_expected_checkpoint_epoch="${CHECKPOINT_EPOCH}" \
    ppr_reference_noise_test=true \
    ppr_reference_noise_scale=1.0 \
    ppr_reference_noise_output_dir="${OUTPUT_DIR}" \
    ppr_reference_noise_overwrite="${OVERWRITE_OUTPUT:-false}" \
    ppr_reference_noise_seeds="${NOISE_SEEDS:-[918273,271828]}" \
    datasets.val.manual_val.limit="${LIMIT:-96}" \
    dataloaders.manual_val.batch_size="${BATCH_SIZE:-12}" \
    "$@"
