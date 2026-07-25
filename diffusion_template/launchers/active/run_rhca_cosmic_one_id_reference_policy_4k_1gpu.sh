#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/launchers/lib/prepare_comet_record.sh"

REFERENCE_POLICY="${REFERENCE_POLICY:-margin40}"
case "${REFERENCE_POLICY}" in
  margin40)
    RUN_NAME_DEFAULT="rhca_cosmic_oneid_margin40_4k"
    BASE_REFERENCE_MODE="full_scene"
    CROP_MARGIN="0.4"
    CONTENT_SIZE="256"
    CANVAS_SIZE="null"
    ;;
  canvas1024)
    RUN_NAME_DEFAULT="rhca_cosmic_oneid_canvas1024_4k"
    BASE_REFERENCE_MODE="cosmic_256"
    CROP_MARGIN="null"
    CONTENT_SIZE="256"
    CANVAS_SIZE="1024"
    ;;
  *)
    echo "Unknown REFERENCE_POLICY=${REFERENCE_POLICY}" >&2
    echo "Expected one of: margin40, canvas1024" >&2
    exit 2
    ;;
esac

export CONFIG_NAME="controlled_identity_reference_policy_rhca"
export RUN_NAME="${RUN_NAME:-${RUN_NAME_DEFAULT}}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-8}"
export COMET_PROJECT="${COMET_PROJECT:-rsrch-jul}"

SPEC_PATH="${ROOT_DIR}/experiments/cosmic_large_adaptation/${RUN_NAME}.json"
prepare_comet_record "${ROOT_DIR}" "${RUN_NAME}" "${SPEC_PATH}"

exec bash "${SCRIPT_DIR}/run_rhca_apr2026_one_id_1gpu.sh" \
  "controlled_factorial.reference_mode=${BASE_REFERENCE_MODE}" \
  "controlled_factorial.reference_crop_margin=${CROP_MARGIN}" \
  "controlled_factorial.reference_content_size=${CONTENT_SIZE}" \
  "controlled_factorial.reference_canvas_size=${CANVAS_SIZE}" \
  "$@"
