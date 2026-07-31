#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/launchers/lib/prepare_comet_record.sh"

: "${RUN_NAME:?Set a unique training run name}"
: "${EXPERIMENT_SPEC_PATH:?Set the experiment JSON path}"
: "${BIG_CELEBS_MANIFEST:?Set the sealed Big Celebs manifest path}"
: "${BIG_CELEBS_IMAGES:?Set the sealed Big Celebs image root}"
: "${BIG_CELEBS_SEAL:?Set the Big Celebs dataset_manifest.json path}"
: "${BIG_CELEBS_EXPECTED_MANIFEST_SHA256:?Set the pinned manifest SHA-256}"

BIG_CELEBS_MIN_FACE_RES="${BIG_CELEBS_MIN_FACE_RES:-192}"
mkdir -p "${ROOT_DIR}/logs/preflight"
python tools/datasets/preflight_big_celebs.py \
  --manifest "${BIG_CELEBS_MANIFEST}" \
  --images-root "${BIG_CELEBS_IMAGES}" \
  --dataset-manifest "${BIG_CELEBS_SEAL}" \
  --expected-sha256 "${BIG_CELEBS_EXPECTED_MANIFEST_SHA256}" \
  --min-face-res "${BIG_CELEBS_MIN_FACE_RES}" \
  --sample-count "${BIG_CELEBS_PREFLIGHT_SAMPLES:-64}" \
  --output "${ROOT_DIR}/logs/preflight/${RUN_NAME}.json"

prepare_comet_record "${ROOT_DIR}" "${RUN_NAME}" "${EXPERIMENT_SPEC_PATH}"

export CONFIG_NAME="big_celebs_rhca_40k"
export TRAIN_EPOCH_LEN="${TRAIN_EPOCH_LEN:-2000}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-20}"
export COMET_PROJECT="${COMET_PROJECT:-jul-comet-large-testing-tr}"

exec bash "${SCRIPT_DIR}/run_rhca_apr2026_one_id_1gpu.sh" \
  "pipeline.pose_adapt_ratio=0.0" \
  "pipeline.ca_mixing_for_face=false" \
  "$@"
