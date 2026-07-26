#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/launchers/lib/prepare_comet_record.sh"

: "${RUN_NAME:?Set RUN_NAME to the unique intervention run name}"
: "${VALIDATION_SOURCE_RUN:?Set VALIDATION_SOURCE_RUN to the completed training run}"
: "${VALIDATION_SOURCE_COMET_KEY:?Set the immutable source Comet key}"
: "${CANONICAL_FULL96_RUN:?Set the completed canonical full-96 run}"
: "${FULL96_BBOX_MANUAL:?Set the sealed full-96 manual bbox path}"
: "${EXPERIMENT_SPEC_PATH:?Set the intervention experiment JSON path}"
: "${POSE_ADAPT_RATIO:?Set target-native face K/V blend in [0, 1]}"

python3 - "${POSE_ADAPT_RATIO}" <<'PY'
import math
import sys

ratio = float(sys.argv[1])
if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
    raise ValueError(f"POSE_ADAPT_RATIO must be in [0, 1], got {ratio}")
PY

export VALIDATION_CHECKPOINT="${ROOT_DIR}/saved/${VALIDATION_SOURCE_RUN}/checkpoint-epoch8.pth"
SOURCE_IMAGES="${ROOT_DIR}/saved/${VALIDATION_SOURCE_RUN}/val_images/manual_val/step_4000_batch_0"

# The intervention deliberately changes pixels, so establish provenance from
# the already-finalized canonical run instead of requiring endpoint equality.
python tools/inference/check_full96_eval_prerequisites.py \
  --project-root "${ROOT_DIR}" \
  --source-run "${VALIDATION_SOURCE_RUN}" \
  --source-comet-key "${VALIDATION_SOURCE_COMET_KEY}" \
  --bbox-manual "${FULL96_BBOX_MANUAL}" \
  --auto-min 95 \
  --require-completed-eval "${CANONICAL_FULL96_RUN}"

export VALIDATION_CHECKPOINT_SHA256
VALIDATION_CHECKPOINT_SHA256="$(sha256sum "${VALIDATION_CHECKPOINT}" | awk '{print $1}')"
export CONFIG_NAME="cosmic_large_adapted_full96_eval_rhca"
export TRAIN_EPOCHS="8"
export COMET_PROJECT="${COMET_PROJECT:-rsrch-jul}"

prepare_comet_record "${ROOT_DIR}" "${RUN_NAME}" "${EXPERIMENT_SPEC_PATH}"

bash "${SCRIPT_DIR}/run_rhca_apr2026_one_id_1gpu.sh" \
  "trainer.from_pretrained=${VALIDATION_CHECKPOINT}" \
  "validation_source_run_name=${VALIDATION_SOURCE_RUN}" \
  "validation_source_comet_key=${VALIDATION_SOURCE_COMET_KEY}" \
  "validation_checkpoint_sha256=${VALIDATION_CHECKPOINT_SHA256}" \
  "pipeline.pose_adapt_ratio=${POSE_ADAPT_RATIO}" \
  "datasets.val.manual_val.bbox_mask_gen=${FULL96_BBOX_MANUAL}"

RECORD="${ROOT_DIR}/saved/${RUN_NAME}/comet_experiment.json"
python tools/comet/comet_experiment.py show "${RECORD}"

EVAL_ROOT="${ROOT_DIR}/saved/${RUN_NAME}/val_images/manual_val"
if [[ "$(find "${EVAL_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'step_4000_batch_*' | wc -l)" -ne 8 ]]; then
  echo "Pose-adapt evaluation did not create eight batches" >&2
  exit 5
fi
if [[ "$(find "${EVAL_ROOT}"/step_4000_batch_* -maxdepth 1 -type f -name '*.png' | wc -l)" -ne 96 ]]; then
  echo "Pose-adapt evaluation did not create 96 PNGs" >&2
  exit 5
fi

COMET_EXPORT_ROOT="${ROOT_DIR}/saved/${RUN_NAME}/comet_step4000_export"
COMET_EXPORT_JSON="${COMET_EXPORT_ROOT}/comet_runs_export.json"
COMET_VERIFIED=false
for attempt in 1 2 3 4 5; do
  if python tools/comet/comet_experiment.py fetch \
      --record "${RECORD}" \
      --step-number 4000 \
      --output-dir "${COMET_EXPORT_ROOT}" \
    && python tools/inference/finalize_full96_eval_record.py \
      --record "${RECORD}" \
      --checkpoint "${VALIDATION_CHECKPOINT}" \
      --bbox-manual "${FULL96_BBOX_MANUAL}" \
      --images-root "${EVAL_ROOT}" \
      --source-images "${SOURCE_IMAGES}" \
      --comet-export "${COMET_EXPORT_JSON}" \
      --intervention-label "pose_adapt_ratio=${POSE_ADAPT_RATIO}" \
      --verify-only; then
    COMET_VERIFIED=true
    break
  fi
  if (( attempt < 5 )); then
    echo "Comet intervention export is incomplete; retrying in 30 seconds" >&2
    sleep 30
  fi
done
if [[ "${COMET_VERIFIED}" != "true" ]]; then
  echo "Comet did not expose the exact 96 intervention images and metrics" >&2
  exit 7
fi

python tools/inference/finalize_full96_eval_record.py \
  --record "${RECORD}" \
  --checkpoint "${VALIDATION_CHECKPOINT}" \
  --bbox-manual "${FULL96_BBOX_MANUAL}" \
  --images-root "${EVAL_ROOT}" \
  --source-images "${SOURCE_IMAGES}" \
  --comet-export "${COMET_EXPORT_JSON}" \
  --intervention-label "pose_adapt_ratio=${POSE_ADAPT_RATIO}"

printf 'POSE_ADAPT_FULL96_COMPLETE run=%s ratio=%s checkpoint_sha256=%s images=96\n' \
  "${RUN_NAME}" \
  "${POSE_ADAPT_RATIO}" \
  "${VALIDATION_CHECKPOINT_SHA256}"
