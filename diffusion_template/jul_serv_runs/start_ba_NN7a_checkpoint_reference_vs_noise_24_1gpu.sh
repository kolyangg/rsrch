#!/usr/bin/env bash
set -euo pipefail

# Deterministic 24/96 RealVis causal matrix. The seed/index manifest is stable
# across runs; override SUBSET_SEED only when intentionally defining a new set.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${1:-}}"
if [[ -z "${CHECKPOINT_PATH}" || ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Pass the NN7a checkpoint-epoch2.pth as argument 1." >&2
  exit 2
fi
if [[ $# -gt 0 && "$1" == "${CHECKPOINT_PATH}" ]]; then shift; fi
CHECKPOINT_PATH="$(cd -- "$(dirname -- "${CHECKPOINT_PATH}")" && pwd)/$(basename -- "${CHECKPOINT_PATH}")"

CHECKPOINT_EPOCH="${CHECKPOINT_EPOCH:-2}"
[[ "${CHECKPOINT_EPOCH}" == "2" ]] || {
  echo "The NN7a post-training approval diagnostic expects epoch 2 (4k)." >&2
  exit 2
}

SUBSET_SIZE="${SUBSET_SIZE:-24}"
SUBSET_SEED="${SUBSET_SEED:-20260722}"
export RUN_FOREGROUND=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MASTER_PORT="${MASTER_PORT:-29673}"
export NUM_EPOCHS=2
export RUN_NAME="${RUN_NAME:-ba_NN7a_4000step_realvis_reference_vs_noise_24}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/ppr_NN7a_4000step_realvis_scale1_reference_vs_noise_subset${SUBSET_SIZE}_seed${SUBSET_SEED}}"

exec bash "${SCRIPT_DIR}/start_ba_NN7a_clean_patch_takeover_up1_1gpu.sh" \
  validation_only=true \
  continue_run=false \
  saved_checkpoint="${CHECKPOINT_PATH}" \
  ppr_checkpoint_require_nonzero=true \
  strict_checkpoint_model_config=true \
  ppr_expected_checkpoint_epoch=2 \
  ppr_reference_noise_test=true \
  ppr_reference_noise_scale=1.0 \
  ppr_reference_noise_output_dir="${OUTPUT_DIR}" \
  ppr_reference_noise_overwrite="${OVERWRITE_OUTPUT:-false}" \
  ppr_reference_noise_seeds="${NOISE_SEEDS:-[918273,271828]}" \
  datasets.val.manual_val.limit=96 \
  +datasets.val.manual_val.subset_size="${SUBSET_SIZE}" \
  +datasets.val.manual_val.subset_seed="${SUBSET_SEED}" \
  dataloaders.manual_val.batch_size="${BATCH_SIZE:-12}" \
  "$@"
