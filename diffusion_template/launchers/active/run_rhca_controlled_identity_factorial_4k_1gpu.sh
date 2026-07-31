#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FACTORIAL_ARM="${FACTORIAL_ARM:-multi_full}"
case "${FACTORIAL_ARM}" in
  multi_full)
    TARGET_MODE="multi"
    REFERENCE_MODE="full_scene"
    ;;
  single_full)
    TARGET_MODE="single"
    REFERENCE_MODE="full_scene"
    ;;
  multi_cosref)
    TARGET_MODE="multi"
    REFERENCE_MODE="cosmic_256"
    ;;
  *)
    echo "Unknown FACTORIAL_ARM=${FACTORIAL_ARM}" >&2
    echo "Expected one of: multi_full, single_full, multi_cosref" >&2
    exit 2
    ;;
esac

export CONFIG_NAME="controlled_identity_factorial_rhca"
export RUN_NAME="${RUN_NAME:-rhca_controlled_identity_factorial_${FACTORIAL_ARM}_4k}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-2}"  # 2 × 2,000 steps = 4,000 total
export COMET_PROJECT="${COMET_PROJECT:-rsrch-jul}"

exec bash "${SCRIPT_DIR}/run_rhca_apr2026_one_id_1gpu.sh" \
  "controlled_factorial.target_mode=${TARGET_MODE}" \
  "controlled_factorial.reference_mode=${REFERENCE_MODE}" \
  "$@"
