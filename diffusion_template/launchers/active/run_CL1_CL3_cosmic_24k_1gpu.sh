#!/usr/bin/env bash
# CL1-CL3: the exact E13 route trained on cosmic_large. The only scientific
# variable is how the reference reaches the branched spatial lane.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/launchers/lib/prepare_comet_record.sh"

: "${RUN_NAME:?Set the unique CL1-CL3 run name}"
: "${CONFIG_NAME:?Set the matching CL1-CL3 config name}"
: "${EXPERIMENT_SPEC_PATH:?Set the matching experiment JSON path}"
: "${COSMIC_LARGE_MANIFEST:?Set the full filtered Cosmic manifest}"
: "${COSMIC_LARGE_ROOT:?Set the Cosmic dataset root containing target/ref trees}"
: "${COMET_API_KEY:?Load COMET_API_KEY from diffusion_template/.env}"
: "${FACE_QUALITY_SCORER_PYTHON:?Set the PyIQA scorer interpreter}"

if [[ "$#" -ne 0 ]]; then
  echo "CL1-CL3 launchers do not accept ad-hoc Hydra overrides." >&2
  exit 2
fi

case "${CONFIG_NAME}" in
  CL1_cosmic_joint_shadow_sa128_sceneref_24k)
    : "${COSMIC_IDENTITY_GROUPS:?CL1 requires the sealed identity grouping}"
    : "${COSMIC_IDENTITY_GROUPS_SHA256:?CL1 requires the identity grouping SHA-256}"
    test -s "${COSMIC_IDENTITY_GROUPS}"
    actual_groups_sha="$(sha256sum "${COSMIC_IDENTITY_GROUPS}" | cut -d' ' -f1)"
    if [[ "${actual_groups_sha}" != "${COSMIC_IDENTITY_GROUPS_SHA256}" ]]; then
      echo "Identity groups hash mismatch: ${actual_groups_sha}" >&2
      exit 73
    fi
    ;;
  CL0_cosmic_joint_shadow_sa128_asis_24k|\
  CL2_cosmic_joint_shadow_sa128_facecanon_24k|\
  CL3_cosmic_joint_shadow_sa128_fmtfix_24k|\
  CL4_cosmic_joint_shadow_sa128_hygiene_24k|\
  CL5_cosmic_joint_shadow_sa128_roiwarp_multiref_24k|\
  CL6_cosmic_joint_shadow_sa128_boundary_24k|\
  CL7_cosmic_joint_shadow_sa128_altloss_24k|\
  CL8_cosmic_joint_shadow_sa128_fullbody_24k|\
  CL9_cosmic_joint_shadow_sa128_refscale_24k|\
  CL10_cosmic_joint_shadow_sa128_refscale_fullbody_24k|\
  CL11_cosmic_joint_shadow_sa128_refscale_multiref_24k|\
  CL12_cosmic_joint_shadow_sa128_refscale_fullbody_multiref_24k|\
  CL13_cosmic_joint_shadow_sa128_refdropout_24k|\
  CL14_cosmic_joint_shadow_sa128_softmask_24k)
    ;;
  *)
    echo "Unapproved CL1-CL3 CONFIG_NAME: ${CONFIG_NAME}" >&2
    exit 2
    ;;
esac

test -s "${COSMIC_LARGE_MANIFEST}"
test -d "${COSMIC_LARGE_ROOT}"
if [[ -n "${COSMIC_REFERENCE_ACCEPT_LIST:-}" ]]; then
  test -s "${COSMIC_REFERENCE_ACCEPT_LIST}"
fi

python tools/validate_CL1_CL3_config.py \
  --config-name "${CONFIG_NAME}" \
  --run-name "${RUN_NAME}" \
  --experiment-spec "${EXPERIMENT_SPEC_PATH}"

mkdir -p "${ROOT_DIR}/logs/preflight"
python tools/datasets/preflight_cosmic_cl.py \
  --config-name "${CONFIG_NAME}" \
  --sample-count "${COSMIC_PREFLIGHT_SAMPLES:-64}" \
  --output "${ROOT_DIR}/logs/preflight/${RUN_NAME}.json"

prepare_comet_record "${ROOT_DIR}" "${RUN_NAME}" "${EXPERIMENT_SPEC_PATH}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HYDRA_FULL_ERROR=1
export ACCELERATE_LOG_LEVEL=error
export TRANSFORMERS_VERBOSITY=error
export DIFFUSERS_VERBOSITY=error
export COMET_DISABLE_AUTO_LOGGING=1
export COMET_LOGGING_CONSOLE=ERROR
export ACCELERATE_NUM_PROCESSES=1

MODEL_OVERRIDES=()
if [[ -n "${PM_PATH:-}" ]]; then
  MODEL_OVERRIDES+=("model.photomaker_path=${PM_PATH}")
fi

set +e
accelerate launch \
  --config_file=src/configs/ddp/accelerate.yaml \
  --num_processes=1 \
  train.py \
  "--config-name=${CONFIG_NAME}" \
  writer=cometml \
  "writer.run_name=${RUN_NAME}" \
  writer.project_name=aug-large-ds \
  "${MODEL_OVERRIDES[@]}"
TRAIN_STATUS=$?
set -e

if [[ "${TRAIN_STATUS}" -ne 0 ]]; then
  echo "Training failed with status ${TRAIN_STATUS}; deferred face quality will not run." >&2
  exit "${TRAIN_STATUS}"
fi

# Training/checkpoints are complete before nonfatal deferred PyIQA scoring.
"${FACE_QUALITY_SCORER_PYTHON}" \
  tools/comet/finalize_deferred_face_quality.py \
  --run-dir "${ROOT_DIR}/saved/${RUN_NAME}" \
  --expected-project aug-large-ds \
  --expected-steps 0,2000,4000,6000,8000,10000,12000,14000,16000,18000,20000,22000,24000 \
  --images-per-step 96 \
  --partition manual_val \
  --scorer-python "${FACE_QUALITY_SCORER_PYTHON}" \
  --device cuda \
  --batch-size 8 \
  --write \
  --upload-per-image-asset \
  --nonfatal
