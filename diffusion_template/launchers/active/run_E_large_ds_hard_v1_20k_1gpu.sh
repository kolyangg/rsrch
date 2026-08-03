#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/launchers/lib/prepare_comet_record.sh"

: "${RUN_NAME:?Set the unique planned run name}"
: "${CONFIG_NAME:?Set one approved August Large Dataset config name}"
: "${EXPERIMENT_SPEC_PATH:?Set the matching experiment JSON path}"
: "${LARGE_DATASET_MANIFEST:?Set the adjusted identity manifest path}"
: "${LARGE_DATASET_IMAGES:?Set the adjusted image root}"

if [[ "$#" -ne 0 ]]; then
  echo "This controlled suite does not accept ad-hoc Hydra overrides." >&2
  exit 2
fi

case "${CONFIG_NAME}" in
  E1_large_ds_truekey_20k|\
  E2_large_ds_branchout_20k|\
  E3_large_ds_roiwarp_20k|\
  E4_large_ds_midup_20k|\
  E5_large_ds_infersteps_20k|\
  E6_large_ds_fp32_20k)
    ;;
  *)
    echo "Unapproved CONFIG_NAME for August Large Dataset suite: ${CONFIG_NAME}" >&2
    exit 2
    ;;
esac

python tools/validate_aug_large_ds_config.py \
  --config-name "${CONFIG_NAME}" \
  --run-name "${RUN_NAME}" \
  --experiment-spec "${EXPERIMENT_SPEC_PATH}"

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

export TRAIN_EPOCH_LEN="2000"
export TRAIN_EPOCHS="10"
export WRITER="cometml"
export COMET_PROJECT="aug-large-ds"
export ACCELERATE_NUM_PROCESSES="1"

exec bash "${SCRIPT_DIR}/run_rhca_apr2026_one_id_1gpu.sh" \
  "pipeline.pose_adapt_ratio=0.0" \
  "pipeline.ca_mixing_for_face=false" \
  "disable_branched_ca=true" \
  "model.ba_enforce_reference_only_hard_route=true"
