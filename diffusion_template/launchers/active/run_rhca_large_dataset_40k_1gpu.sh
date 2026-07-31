#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/launchers/lib/prepare_comet_record.sh"

: "${RUN_NAME:?Set a unique training run name}"
: "${EXPERIMENT_SPEC_PATH:?Set the experiment JSON path}"
: "${LARGE_DATASET_MANIFEST:?Set the adjusted identity manifest path}"
: "${LARGE_DATASET_IMAGES:?Set the adjusted image root}"

mkdir -p "${ROOT_DIR}/logs/preflight"
preflight_args=(
  tools/datasets/preflight_large_dataset.py
  --manifest "${LARGE_DATASET_MANIFEST}"
  --images-root "${LARGE_DATASET_IMAGES}"
  --sample-count "${LARGE_DATASET_PREFLIGHT_SAMPLES:-64}"
  --output "${ROOT_DIR}/logs/preflight/${RUN_NAME}.json"
)
if [[ -n "${LARGE_DATASET_SEAL:-}" ]]; then
  preflight_args+=(--dataset-manifest "${LARGE_DATASET_SEAL}")
fi
python "${preflight_args[@]}"

prepare_comet_record "${ROOT_DIR}" "${RUN_NAME}" "${EXPERIMENT_SPEC_PATH}"

export CONFIG_NAME="large_dataset_rhca_40k"
export TRAIN_EPOCHS="20"
export COMET_PROJECT="${COMET_PROJECT:-jul-comet-large-testing-tr}"

exec bash "${SCRIPT_DIR}/run_rhca_apr2026_one_id_1gpu.sh" \
  "pipeline.pose_adapt_ratio=0.0" \
  "pipeline.ca_mixing_for_face=false" \
  "$@"
