#!/usr/bin/env bash
# Unified one-GPU Serv entry point. CONFIG_NAME selects all scientific behavior.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

: "${RUN_NAME:?Set a new, unique run name}"
: "${CONFIG_NAME:?Select a config listed by tools/validate_clean_full_config.py --list}"
: "${COMET_API_KEY:?Load COMET_API_KEY from diffusion_template/.env}"
: "${FACE_QUALITY_SCORER_PYTHON:?Set the PyIQA scorer interpreter}"
: "${SUBJECT_V2_ID_EMBEDS:?Set the sealed subject-v2 embedding file}"

if [[ "$#" -ne 0 ]]; then
  echo "clean_full rejects ad-hoc Hydra overrides; edit or select a reviewed config." >&2
  exit 2
fi
if [[ "${COMET_PROJECT:-aug-large-ds}" != "aug-large-ds" ]]; then
  echo "clean_full is pinned to Comet project aug-large-ds." >&2
  exit 2
fi
export COMET_PROJECT=aug-large-ds

PYTHON_BIN="${PYTHON_BIN:-python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
VALIDATOR="${ROOT_DIR}/tools/validate_clean_full_config.py"
PREFLIGHT_DIR="${ROOT_DIR}/logs/preflight"
mkdir -p "${PREFLIGHT_DIR}"

DATASET="$("${PYTHON_BIN}" "${VALIDATOR}" --config-name "${CONFIG_NAME}" --field dataset)"
VALIDATION_STEPS="$("${PYTHON_BIN}" "${VALIDATOR}" --config-name "${CONFIG_NAME}" --field validation_steps)"
VALIDATION_ONLY="$("${PYTHON_BIN}" "${VALIDATOR}" --config-name "${CONFIG_NAME}" --field validation_only)"

test -s "${SUBJECT_V2_ID_EMBEDS}"
if [[ "$(sha256sum "${SUBJECT_V2_ID_EMBEDS}" | cut -d' ' -f1)" != \
  "e0d36212ad350db8252c4805acf46aa4c90289603d460584dc7692066712b465" ]]; then
  echo "SUBJECT_V2_ID_EMBEDS does not match the sealed fixed-96 asset." >&2
  exit 3
fi

case "${DATASET}" in
  cosmic_large_adapted)
    : "${COSMIC_LARGE_MANIFEST:?Set the filtered Cosmic manifest}"
    : "${COSMIC_LARGE_ROOT:?Set the Cosmic image root}"
    test -s "${COSMIC_LARGE_MANIFEST}"
    test -d "${COSMIC_LARGE_ROOT}"
    "${PYTHON_BIN}" tools/datasets/preflight_cosmic_cl.py \
      --config-name "${CONFIG_NAME}" \
      --sample-count "${COSMIC_PREFLIGHT_SAMPLES:-64}" \
      --output "${PREFLIGHT_DIR}/${RUN_NAME}_cosmic.json"
    ;;
  large_dataset)
    : "${LARGE_DATASET_MANIFEST:?Set the adjusted Large Dataset manifest}"
    : "${LARGE_DATASET_IMAGES:?Set the adjusted Large Dataset image root}"
    "${PYTHON_BIN}" tools/datasets/preflight_large_dataset.py \
      --manifest "${LARGE_DATASET_MANIFEST}" \
      --images-root "${LARGE_DATASET_IMAGES}" \
      --sample-count "${LARGE_DATASET_PREFLIGHT_SAMPLES:-64}" \
      --output "${PREFLIGHT_DIR}/${RUN_NAME}_large.json"
    ;;
  big_celebs|bc_e13_ds1|bc_e13_ds2|bc_e13_ds3)
    : "${BIG_CELEBS_MANIFEST:?Set the sealed BigCelebs manifest}"
    : "${BIG_CELEBS_IMAGES:?Set the sealed BigCelebs image root}"
    : "${BIG_CELEBS_SEAL:?Set the BigCelebs dataset_manifest.json}"
    : "${BIG_CELEBS_DOWNLOAD_LOG:?Set the completed BigCelebs download log}"
    : "${BIG_CELEBS_EXPECTED_MANIFEST_SHA256:?Pin the BigCelebs manifest SHA-256}"
    if ! grep -qF "BIGCELEBS_V2_DOWNLOAD_COMPLETE" "${BIG_CELEBS_DOWNLOAD_LOG}"; then
      echo "BigCelebs v2 download is incomplete." >&2
      exit 3
    fi
    "${PYTHON_BIN}" tools/datasets/preflight_big_celebs.py \
      --manifest "${BIG_CELEBS_MANIFEST}" \
      --images-root "${BIG_CELEBS_IMAGES}" \
      --dataset-manifest "${BIG_CELEBS_SEAL}" \
      --expected-sha256 "${BIG_CELEBS_EXPECTED_MANIFEST_SHA256}" \
      --min-face-res "${BIG_CELEBS_MIN_FACE_RES:-192}" \
      --sample-count "${BIG_CELEBS_PREFLIGHT_SAMPLES:-64}" \
      --output "${PREFLIGHT_DIR}/${RUN_NAME}_big.json"
    if [[ "${DATASET}" == bc_e13_ds* ]]; then
      MODE="${DATASET#bc_e13_}"
      : "${BC_E13_SCHEDULE:?Set the sealed 48k schedule}"
      : "${BC_E13_SCHEDULE_SUMMARY:?Set the sealed schedule summary}"
      : "${BC_E13_EXPECTED_SCHEDULE_SHA256:?Pin the schedule SHA-256}"
      SCHEDULE_ARGS=(
        --mode "${MODE}"
        --schedule "${BC_E13_SCHEDULE}"
        --summary "${BC_E13_SCHEDULE_SUMMARY}"
        --expected-schedule-sha256 "${BC_E13_EXPECTED_SCHEDULE_SHA256}"
        --big-manifest "${BIG_CELEBS_MANIFEST}"
        --big-images "${BIG_CELEBS_IMAGES}"
        --big-manifest-sha256 "${BIG_CELEBS_EXPECTED_MANIFEST_SHA256}"
        --sample-count "${BC_E13_PREFLIGHT_SAMPLES:-64}"
        --output "${PREFLIGHT_DIR}/${RUN_NAME}_schedule.json"
      )
      if [[ "${MODE}" == "ds3" ]]; then
        : "${LARGE_DATASET_MANIFEST:?ds3 requires the Large Dataset manifest}"
        : "${LARGE_DATASET_IMAGES:?ds3 requires the Large Dataset image root}"
        : "${LARGE_DATASET_EXPECTED_MANIFEST_SHA256:?ds3 requires its manifest SHA-256}"
        "${PYTHON_BIN}" tools/datasets/preflight_large_dataset.py \
          --manifest "${LARGE_DATASET_MANIFEST}" \
          --images-root "${LARGE_DATASET_IMAGES}" \
          --sample-count "${LARGE_DATASET_PREFLIGHT_SAMPLES:-64}" \
          --output "${PREFLIGHT_DIR}/${RUN_NAME}_large.json"
        SCHEDULE_ARGS+=(
          --large-manifest "${LARGE_DATASET_MANIFEST}"
          --large-images "${LARGE_DATASET_IMAGES}"
          --large-manifest-sha256 "${LARGE_DATASET_EXPECTED_MANIFEST_SHA256}"
        )
      fi
      "${PYTHON_BIN}" tools/datasets/preflight_bc_e13_dataset_schedule.py "${SCHEDULE_ARGS[@]}"
    fi
    ;;
  *)
    echo "No clean_full preflight is registered for dataset ${DATASET}." >&2
    exit 2
    ;;
esac

# Dataset audits happen before the first persistent run or Comet mutation.
RUN_RECORD="${ROOT_DIR}/saved/${RUN_NAME}/comet_experiment.json"
"${PYTHON_BIN}" "${VALIDATOR}" \
  --config-name "${CONFIG_NAME}" \
  --run-name "${RUN_NAME}" \
  --write-run-record "${RUN_RECORD}" >/dev/null

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HYDRA_FULL_ERROR=1 ACCELERATE_LOG_LEVEL=error TRANSFORMERS_VERBOSITY=error
export DIFFUSERS_VERBOSITY=error COMET_DISABLE_AUTO_LOGGING=1 COMET_LOGGING_CONSOLE=ERROR
export ACCELERATE_NUM_PROCESSES=1

MODEL_OVERRIDES=("metrics.id_sim_subject_v2.id_embeds_pth=${SUBJECT_V2_ID_EMBEDS}")
if [[ -n "${PM_PATH:-}" ]]; then
  test -s "${PM_PATH}"
  MODEL_OVERRIDES+=("model.photomaker_path=${PM_PATH}")
fi

set +e
"${ACCELERATE_BIN}" launch \
  --config_file=src/configs/ddp/accelerate.yaml \
  --num_processes=1 \
  train.py \
  "--config-name=${CONFIG_NAME}" \
  writer=cometml \
  "writer.run_name=${RUN_NAME}" \
  writer.project_name=aug-large-ds \
  "${MODEL_OVERRIDES[@]}" &
TRAIN_PID=$!
set -e

COMET_READY=0
for _ in $(seq 1 300); do
  if [[ -s "${RUN_RECORD}" ]] && "${PYTHON_BIN}" - "${RUN_RECORD}" <<'PY'
import json
import sys

key = (json.load(open(sys.argv[1], encoding="utf-8")).get("comet") or {}).get("experiment_key")
raise SystemExit(0 if isinstance(key, str) and len(key) == 32 else 1)
PY
  then
    COMET_READY=1
    echo "COMET_STARTUP_VERIFIED ${RUN_RECORD}"
    break
  fi
  if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
    wait "${TRAIN_PID}"
    exit $?
  fi
  sleep 2
done
if [[ "${COMET_READY}" -ne 1 ]]; then
  echo "Comet immutable key was not registered within 10 minutes." >&2
  kill "${TRAIN_PID}" 2>/dev/null || true
  wait "${TRAIN_PID}" || true
  exit 78
fi

set +e
wait "${TRAIN_PID}"
TRAIN_STATUS=$?
set -e
if [[ "${TRAIN_STATUS}" -ne 0 ]]; then
  echo "Run failed with status ${TRAIN_STATUS}; deferred face quality was not started." >&2
  exit "${TRAIN_STATUS}"
fi

FINALIZE_ARGS=(
  --run-dir "${ROOT_DIR}/saved/${RUN_NAME}"
  --expected-project aug-large-ds
  --expected-steps "${VALIDATION_STEPS}"
  --images-per-step 96
  --partition manual_val
  --scorer-python "${FACE_QUALITY_SCORER_PYTHON}"
  --device cuda
  --batch-size 8
  --write
  --upload-per-image-asset
)
if [[ "${VALIDATION_ONLY}" != "True" ]]; then
  FINALIZE_ARGS+=(--nonfatal)
fi
"${FACE_QUALITY_SCORER_PYTHON}" tools/comet/finalize_deferred_face_quality.py \
  "${FINALIZE_ARGS[@]}"
