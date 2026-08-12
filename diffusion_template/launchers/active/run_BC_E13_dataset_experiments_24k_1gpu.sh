#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/launchers/lib/prepare_comet_record.sh"

: "${RUN_NAME:?Set the exact BC_E13_ds1-ds3 run name}"
: "${CONFIG_NAME:?Set the exact BC_E13_ds1-ds3 config name}"
: "${BC_E13_DATASET_MODE:?Set ds1, ds2, or ds3}"
: "${EXPERIMENT_SPEC_PATH:?Set the matching immutable experiment JSON}"
: "${BC_E13_SCHEDULE:?Set the sealed 48k schedule path}"
: "${BC_E13_SCHEDULE_SUMMARY:?Set the sealed schedule summary path}"
: "${BC_E13_EXPECTED_SCHEDULE_SHA256:?Set the sealed schedule SHA-256}"
: "${BIG_CELEBS_MANIFEST:?Set the BigCelebs v2 manifest}"
: "${BIG_CELEBS_IMAGES:?Set the BigCelebs v2 image root}"
: "${BIG_CELEBS_SEAL:?Set the BigCelebs dataset_manifest.json}"
: "${BIG_CELEBS_DOWNLOAD_LOG:?Set the BigCelebs completed download log}"
: "${BIG_CELEBS_EXPECTED_MANIFEST_SHA256:?Pin the BigCelebs manifest SHA-256}"
: "${COMET_API_KEY:?Load COMET_API_KEY from diffusion_template/.env}"
: "${FACE_QUALITY_SCORER_PYTHON:?Set the PyIQA scorer interpreter}"

if [[ "$#" -ne 0 ]]; then
  echo "BC_E13 dataset launchers reject ad-hoc Hydra overrides." >&2
  exit 2
fi
case "${RUN_NAME}|${CONFIG_NAME}|${BC_E13_DATASET_MODE}" in
  "BC_E13_ds1_repeatdepth_balanced_24k_full96_r1|BC_E13_ds1_repeatdepth_balanced_24k|ds1"|\
  "BC_E13_ds2_scene_target_canonical_ref_24k_full96_r1|BC_E13_ds2_scene_target_canonical_ref_24k|ds2"|\
  "BC_E13_ds3_large_anchor_2to1_24k_full96_r1|BC_E13_ds3_large_anchor_2to1_24k|ds3") ;;
  *) echo "Unapproved BC_E13 dataset run/config/mode tuple." >&2; exit 2 ;;
esac
if [[ "${COMET_PROJECT:-aug-large-ds}" != "aug-large-ds" ]]; then
  echo "BC_E13 dataset experiments must use aug-large-ds." >&2
  exit 2
fi
export COMET_PROJECT=aug-large-ds

if ! grep -qF "BIGCELEBS_V2_DOWNLOAD_COMPLETE" "${BIG_CELEBS_DOWNLOAD_LOG}"; then
  echo "BigCelebs v2 download is incomplete." >&2
  exit 3
fi

python tools/validate_BC_E13_dataset_experiments.py \
  --config-name "${CONFIG_NAME}" \
  --run-name "${RUN_NAME}" \
  --experiment-spec "${EXPERIMENT_SPEC_PATH}"

mkdir -p "${ROOT_DIR}/logs/preflight"
python tools/datasets/preflight_big_celebs.py \
  --manifest "${BIG_CELEBS_MANIFEST}" \
  --images-root "${BIG_CELEBS_IMAGES}" \
  --dataset-manifest "${BIG_CELEBS_SEAL}" \
  --expected-sha256 "${BIG_CELEBS_EXPECTED_MANIFEST_SHA256}" \
  --min-face-res 192 \
  --sample-count "${BIG_CELEBS_PREFLIGHT_SAMPLES:-64}" \
  --output "${ROOT_DIR}/logs/preflight/${RUN_NAME}_big_source.json"

SCHEDULE_PREFLIGHT_ARGS=(
  --mode "${BC_E13_DATASET_MODE}"
  --schedule "${BC_E13_SCHEDULE}"
  --summary "${BC_E13_SCHEDULE_SUMMARY}"
  --expected-schedule-sha256 "${BC_E13_EXPECTED_SCHEDULE_SHA256}"
  --big-manifest "${BIG_CELEBS_MANIFEST}"
  --big-images "${BIG_CELEBS_IMAGES}"
  --big-manifest-sha256 "${BIG_CELEBS_EXPECTED_MANIFEST_SHA256}"
  --sample-count "${BC_E13_PREFLIGHT_SAMPLES:-64}"
  --output "${ROOT_DIR}/logs/preflight/${RUN_NAME}_schedule.json"
)
if [[ "${BC_E13_DATASET_MODE}" == "ds3" ]]; then
  : "${LARGE_DATASET_MANIFEST:?ds3 requires the Large Dataset manifest}"
  : "${LARGE_DATASET_IMAGES:?ds3 requires the Large Dataset image root}"
  : "${LARGE_DATASET_EXPECTED_MANIFEST_SHA256:?ds3 requires the Large manifest SHA-256}"
  python tools/datasets/preflight_large_dataset.py \
    --manifest "${LARGE_DATASET_MANIFEST}" \
    --images-root "${LARGE_DATASET_IMAGES}" \
    --sample-count "${LARGE_DATASET_PREFLIGHT_SAMPLES:-64}" \
    --output "${ROOT_DIR}/logs/preflight/${RUN_NAME}_large_source.json"
  SCHEDULE_PREFLIGHT_ARGS+=(
    --large-manifest "${LARGE_DATASET_MANIFEST}"
    --large-images "${LARGE_DATASET_IMAGES}"
    --large-manifest-sha256 "${LARGE_DATASET_EXPECTED_MANIFEST_SHA256}"
  )
fi
python tools/datasets/preflight_bc_e13_dataset_schedule.py \
  "${SCHEDULE_PREFLIGHT_ARGS[@]}"

# The complete source/schedule/decode audits above intentionally precede the
# only call that can create a Comet experiment.
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
