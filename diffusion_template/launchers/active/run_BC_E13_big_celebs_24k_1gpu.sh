#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/launchers/lib/prepare_comet_record.sh"

EXPECTED_RUN_NAME="BC_E13_big_celebs_joint_shadow_sa128_24k_full96_r1"
EXPECTED_CONFIG_NAME="BC_E13_big_celebs_joint_shadow_sa128_24k"

: "${RUN_NAME:?Set the unique BC_E13 BigCelebs run name}"
: "${CONFIG_NAME:?Set the BC_E13 BigCelebs config name}"
: "${EXPERIMENT_SPEC_PATH:?Set the matching experiment JSON path}"
: "${BIG_CELEBS_MANIFEST:?Set the sealed BigCelebs manifest path}"
: "${BIG_CELEBS_IMAGES:?Set the sealed BigCelebs image root}"
: "${BIG_CELEBS_SEAL:?Set the BigCelebs dataset_manifest.json path}"
: "${BIG_CELEBS_DOWNLOAD_LOG:?Set the BigCelebs download log path}"
: "${BIG_CELEBS_EXPECTED_MANIFEST_SHA256:?Pin the manifest SHA-256}"
: "${COMET_API_KEY:?Load COMET_API_KEY from diffusion_template/.env}"
: "${FACE_QUALITY_SCORER_PYTHON:?Set the PyIQA scorer interpreter}"

if [[ "$#" -ne 0 ]]; then
  echo "The BC_E13 BigCelebs launcher does not accept ad-hoc Hydra overrides." >&2
  exit 2
fi
if [[ "${RUN_NAME}" != "${EXPECTED_RUN_NAME}" ]]; then
  echo "Unexpected RUN_NAME: ${RUN_NAME}" >&2
  exit 2
fi
if [[ "${CONFIG_NAME}" != "${EXPECTED_CONFIG_NAME}" ]]; then
  echo "Unexpected CONFIG_NAME: ${CONFIG_NAME}" >&2
  exit 2
fi
if [[ "${COMET_PROJECT:-aug-large-ds}" != "aug-large-ds" ]]; then
  echo "BC_E13 BigCelebs must use Comet project aug-large-ds." >&2
  exit 2
fi
export COMET_PROJECT=aug-large-ds

# 08 Aug 2026 - A partial Serv extraction already contains release metadata;
# require the downloader's terminal success marker before any Comet mutation.
if ! grep -qF "BIGCELEBS_V2_DOWNLOAD_COMPLETE" "${BIG_CELEBS_DOWNLOAD_LOG}"; then
  echo "BigCelebs v2 download is not complete: ${BIG_CELEBS_DOWNLOAD_LOG}" >&2
  exit 3
fi

python tools/validate_BC_E13_big_celebs_config.py \
  --config-name "${CONFIG_NAME}" \
  --run-name "${RUN_NAME}" \
  --experiment-spec "${EXPERIMENT_SPEC_PATH}"

mkdir -p "${ROOT_DIR}/logs/preflight"
python tools/datasets/preflight_big_celebs.py \
  --manifest "${BIG_CELEBS_MANIFEST}" \
  --images-root "${BIG_CELEBS_IMAGES}" \
  --dataset-manifest "${BIG_CELEBS_SEAL}" \
  --expected-sha256 "${BIG_CELEBS_EXPECTED_MANIFEST_SHA256}" \
  --min-face-res "${BIG_CELEBS_MIN_FACE_RES:-192}" \
  --sample-count "${BIG_CELEBS_PREFLIGHT_SAMPLES:-64}" \
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
