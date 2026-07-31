#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/launchers/lib/prepare_comet_record.sh"

: "${COSMIC_INITIAL_IMAGES_ROOT:?Set the root containing LAION-5B-Filtered-Large}"
: "${DATASET_POLICY_ARM:?Set the registered dataset-policy arm}"

COSMIC_INITIAL_METADATA="${COSMIC_INITIAL_METADATA:-${ROOT_DIR}/../dataset_full/cosmic_large_alldata.json}"
COSMIC_INITIAL_CAPTIONS="${COSMIC_INITIAL_CAPTIONS:-${ROOT_DIR}/../dataset_full/all_texts_cosmic_large_by_qween3_trigger_word.json}"
TOPK_TEMPERATURE="0.05"
case "${DATASET_POLICY_ARM}" in
  baseline_self)
    RUN_NAME_DEFAULT="rhca_cosmic_initial_selfref_4k_baseline"
    REFERENCE_MODE="self"
    MIN_FACE_RES="0"
    ;;
  distinct_uniform)
    RUN_NAME_DEFAULT="rhca_cosmic_initial_distinct_uniform_4k"
    REFERENCE_MODE="uniform"
    MIN_FACE_RES="0"
    ;;
  distinct_highest)
    RUN_NAME_DEFAULT="rhca_cosmic_initial_distinct_highest_4k"
    REFERENCE_MODE="highest_score"
    MIN_FACE_RES="0"
    ;;
  distinct_top3_softmax)
    RUN_NAME_DEFAULT="rhca_cosmic_initial_distinct_top3softmax_4k"
    REFERENCE_MODE="top3_softmax"
    MIN_FACE_RES="0"
    ;;
  target_min256)
    RUN_NAME_DEFAULT="rhca_cosmic_initial_selfref_minface256_4k"
    REFERENCE_MODE="self"
    MIN_FACE_RES="256"
    ;;
  *)
    echo "Unknown DATASET_POLICY_ARM=${DATASET_POLICY_ARM}" >&2
    echo "Expected baseline_self, distinct_uniform, distinct_highest," >&2
    echo "distinct_top3_softmax, or target_min256" >&2
    exit 2
    ;;
esac

if [[ "${REFERENCE_MODE}" != "self" ]]; then
  : "${COSMIC_LARGE_MANIFEST:?Distinct-reference arms require the 59k candidate manifest}"
fi

export CONFIG_NAME="cosmic_large_initial_usage_rhca"
export RUN_NAME="${RUN_NAME:-${RUN_NAME_DEFAULT}}"
export TRAIN_EPOCHS="2"
export COMET_PROJECT="${COMET_PROJECT:-rsrch-jul}"

PREFLIGHT_ARGS=(
  --metadata "${COSMIC_INITIAL_METADATA}"
  --captions "${COSMIC_INITIAL_CAPTIONS}"
  --images-root "${COSMIC_INITIAL_IMAGES_ROOT}"
  --reference-mode "${REFERENCE_MODE}"
  --min-face-res "${MIN_FACE_RES}"
  --topk-temperature "${TOPK_TEMPERATURE}"
  --sample-count "${COSMIC_PREFLIGHT_SAMPLES:-64}"
  --output "${ROOT_DIR}/logs/preflight/${RUN_NAME}.json"
)
if [[ "${REFERENCE_MODE}" != "self" ]]; then
  PREFLIGHT_ARGS+=(--candidate-manifest "${COSMIC_LARGE_MANIFEST}")
fi
mkdir -p "${ROOT_DIR}/logs/preflight"
python tools/datasets/preflight_cosmic_large_initial_usage.py \
  "${PREFLIGHT_ARGS[@]}"

SPEC_PATH="${EXPERIMENT_SPEC_PATH:-${ROOT_DIR}/experiments/cosmic_large_dataset_usage/${RUN_NAME}.json}"
prepare_comet_record "${ROOT_DIR}" "${RUN_NAME}" "${SPEC_PATH}"

exec bash "${SCRIPT_DIR}/run_rhca_apr2026_one_id_1gpu.sh" \
  "dataset_policy.reference_mode=${REFERENCE_MODE}" \
  "dataset_policy.min_face_res=${MIN_FACE_RES}" \
  "dataset_policy.topk_temperature=${TOPK_TEMPERATURE}" \
  "pipeline.pose_adapt_ratio=0.0" \
  "pipeline.ca_mixing_for_face=false" \
  "$@"
