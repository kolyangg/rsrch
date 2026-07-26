#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/launchers/lib/prepare_comet_record.sh"

: "${COSMIC_LARGE_MANIFEST:?Set COSMIC_LARGE_MANIFEST to the full filtered JSON}"
: "${COSMIC_LARGE_ROOT:?Set COSMIC_LARGE_ROOT to the directory containing target/ref trees}"

EXPERIMENT_ARM="${EXPERIMENT_ARM:-crop20_legacy_4k}"
POSE_ADAPT_RATIO="0.0"
VALIDATION_POSE_ADAPT_RATIO="null"
case "${EXPERIMENT_ARM}" in
  crop20_legacy_4k)
    RUN_NAME_DEFAULT="rhca_cosmic_full_crop20_legacy_4k"
    CROP_MARGIN="0.2"
    CONTENT_SIZE="256"
    CANVAS_SIZE="null"
    PROMPT_MODE="legacy"
    PROMPT_MAX_WORDS="null"
    TRAIN_EPOCHS_DEFAULT="8"
    ;;
  crop20_posefirst_4k)
    RUN_NAME_DEFAULT="rhca_cosmic_full_crop20_posefirst_4k"
    CROP_MARGIN="0.2"
    CONTENT_SIZE="256"
    CANVAS_SIZE="null"
    PROMPT_MODE="pose_first"
    PROMPT_MAX_WORDS="55"
    TRAIN_EPOCHS_DEFAULT="8"
    ;;
  crop40_posefirst_4k)
    RUN_NAME_DEFAULT="rhca_cosmic_full_crop40_posefirst_4k"
    CROP_MARGIN="0.4"
    CONTENT_SIZE="256"
    CANVAS_SIZE="null"
    PROMPT_MODE="pose_first"
    PROMPT_MAX_WORDS="55"
    TRAIN_EPOCHS_DEFAULT="8"
    ;;
  crop20_posefirst_par100_4k)
    RUN_NAME_DEFAULT="rhca_cosmic_full_crop20_posefirst_par100_4k_r2"
    CROP_MARGIN="0.2"
    CONTENT_SIZE="256"
    CANVAS_SIZE="null"
    PROMPT_MODE="pose_first"
    PROMPT_MAX_WORDS="55"
    POSE_ADAPT_RATIO="1.0"
    TRAIN_EPOCHS_DEFAULT="8"
    ;;
  crop20_posefirst_par100_20k)
    # 26 Jul 2026 - Keep the promoted target-native training policy behind an
    # explicit arm so all historical selectors retain their original routing.
    RUN_NAME_DEFAULT="rhca_cosmic_full_crop20_posefirst_par100_20k"
    CROP_MARGIN="0.2"
    CONTENT_SIZE="256"
    CANVAS_SIZE="null"
    PROMPT_MODE="pose_first"
    PROMPT_MAX_WORDS="55"
    POSE_ADAPT_RATIO="1.0"
    VALIDATION_POSE_ADAPT_RATIO="1.0"
    TRAIN_EPOCHS_DEFAULT="40"
    ;;
  canvas1024_posefirst_4k)
    RUN_NAME_DEFAULT="rhca_cosmic_full_canvas1024_posefirst_4k"
    CROP_MARGIN="0.2"
    CONTENT_SIZE="256"
    CANVAS_SIZE="1024"
    PROMPT_MODE="pose_first"
    PROMPT_MAX_WORDS="55"
    TRAIN_EPOCHS_DEFAULT="8"
    ;;
  crop20_posefirst_20k)
    RUN_NAME_DEFAULT="rhca_cosmic_full_crop20_posefirst_20k"
    CROP_MARGIN="0.2"
    CONTENT_SIZE="256"
    CANVAS_SIZE="null"
    PROMPT_MODE="pose_first"
    PROMPT_MAX_WORDS="55"
    TRAIN_EPOCHS_DEFAULT="40"
    ;;
  crop20_posefirst_trainpar0_valpar100_20k)
    RUN_NAME_DEFAULT="rhca_cosmic_full_crop20_posefirst_trainpar0_valpar100_20k"
    CROP_MARGIN="0.2"
    CONTENT_SIZE="256"
    CANVAS_SIZE="null"
    PROMPT_MODE="pose_first"
    PROMPT_MAX_WORDS="55"
    VALIDATION_POSE_ADAPT_RATIO="1.0"
    TRAIN_EPOCHS_DEFAULT="40"
    ;;
  *)
    echo "Unknown EXPERIMENT_ARM=${EXPERIMENT_ARM}" >&2
    echo "Expected: crop20_legacy_4k, crop20_posefirst_4k," >&2
    echo "          crop40_posefirst_4k," >&2
    echo "          crop20_posefirst_par100_4k," >&2
    echo "          crop20_posefirst_par100_20k," >&2
    echo "          canvas1024_posefirst_4k, crop20_posefirst_20k," >&2
    echo "          crop20_posefirst_trainpar0_valpar100_20k" >&2
    exit 2
    ;;
esac

export CONFIG_NAME="cosmic_large_adapted_rhca"
export RUN_NAME="${RUN_NAME:-${RUN_NAME_DEFAULT}}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-${TRAIN_EPOCHS_DEFAULT}}"
export COMET_PROJECT="${COMET_PROJECT:-rsrch-jul}"

mkdir -p "${ROOT_DIR}/logs/preflight"
python tools/datasets/preflight_cosmic_large_adapted.py \
  --manifest "${COSMIC_LARGE_MANIFEST}" \
  --dataset-root "${COSMIC_LARGE_ROOT}" \
  --sample-count "${COSMIC_PREFLIGHT_SAMPLES:-64}" \
  --crop-margin "${CROP_MARGIN}" \
  --content-size "${CONTENT_SIZE}" \
  --canvas-size "${CANVAS_SIZE}" \
  --prompt-mode "${PROMPT_MODE}" \
  --prompt-max-words "${PROMPT_MAX_WORDS}" \
  --output "${ROOT_DIR}/logs/preflight/${RUN_NAME}.json"

SPEC_PATH="${EXPERIMENT_SPEC_PATH:-${ROOT_DIR}/experiments/cosmic_large_adaptation/${RUN_NAME}.json}"
prepare_comet_record "${ROOT_DIR}" "${RUN_NAME}" "${SPEC_PATH}"

TRAINER_OVERRIDES=()
if [[ "${VALIDATION_POSE_ADAPT_RATIO}" != "null" ]]; then
  TRAINER_OVERRIDES+=(
    "++trainer.validation_pose_adapt_ratio=${VALIDATION_POSE_ADAPT_RATIO}"
  )
fi

exec bash "${SCRIPT_DIR}/run_rhca_apr2026_one_id_1gpu.sh" \
  "cosmic_reference.crop_margin=${CROP_MARGIN}" \
  "cosmic_reference.content_size=${CONTENT_SIZE}" \
  "cosmic_reference.canvas_size=${CANVAS_SIZE}" \
  "cosmic_reference.prompt_mode=${PROMPT_MODE}" \
  "cosmic_reference.prompt_max_words=${PROMPT_MAX_WORDS}" \
  "pipeline.pose_adapt_ratio=${POSE_ADAPT_RATIO}" \
  "pipeline.ca_mixing_for_face=false" \
  "${TRAINER_OVERRIDES[@]}" \
  "$@"
