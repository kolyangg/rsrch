#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/launchers/lib/prepare_comet_record.sh"

: "${RUN_NAME:?Set a unique training run name containing bs4}"
: "${EXPERIMENT_SPEC_PATH:?Set the experiment JSON path}"
: "${BIG_CELEBS_MANIFEST:?Set the sealed BigCelebs manifest path}"
: "${BIG_CELEBS_IMAGES:?Set the sealed BigCelebs image root}"
: "${BIG_CELEBS_SEAL:?Set the BigCelebs dataset_manifest.json path}"
: "${BIG_CELEBS_SAMPLING_PLAN:?Set the pinned batch-4 sampling-plan JSONL path}"
: "${BIG_CELEBS_SAMPLING_PLAN_MANIFEST:?Set the batch-4 sampling-plan manifest path}"
: "${BIG_CELEBS_EXPECTED_MANIFEST_SHA256:?Pin the BigCelebs manifest SHA-256}"
: "${BIG_CELEBS_EXPECTED_SAMPLING_PLAN_SHA256:?Pin the batch-4 plan SHA-256}"

if [[ "${RUN_NAME}" != *bs4* ]]; then
  echo "The batch-4 RUN_NAME must contain 'bs4'" >&2
  exit 2
fi
BIG_CELEBS_SCHEDULE_START_STEP="${BIG_CELEBS_SCHEDULE_START_STEP:-0}"
BIG_CELEBS_GLOBAL_BATCH_SIZE="${BIG_CELEBS_GLOBAL_BATCH_SIZE:-4}"
if ! [[ "${BIG_CELEBS_SCHEDULE_START_STEP}" =~ ^[0-9]+$ ]]; then
  echo "BIG_CELEBS_SCHEDULE_START_STEP must be a non-negative integer" >&2
  exit 2
fi
if [[ "${BIG_CELEBS_GLOBAL_BATCH_SIZE}" != "4" ]]; then
  echo "This launcher requires BIG_CELEBS_GLOBAL_BATCH_SIZE=4" >&2
  exit 2
fi
export BIG_CELEBS_SCHEDULE_START_ROW="$((BIG_CELEBS_SCHEDULE_START_STEP * 4))"

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

python tools/datasets/preflight_big_celebs_schedule.py \
  --manifest "${BIG_CELEBS_MANIFEST}" \
  --expected-manifest-sha256 "${BIG_CELEBS_EXPECTED_MANIFEST_SHA256}" \
  --images-root "${BIG_CELEBS_IMAGES}" \
  --plan "${BIG_CELEBS_SAMPLING_PLAN}" \
  --plan-manifest "${BIG_CELEBS_SAMPLING_PLAN_MANIFEST}" \
  --expected-plan-sha256 "${BIG_CELEBS_EXPECTED_SAMPLING_PLAN_SHA256}" \
  --schedule-start-step "${BIG_CELEBS_SCHEDULE_START_STEP}" \
  --schedule-start-row "${BIG_CELEBS_SCHEDULE_START_ROW}" \
  --sample-count "${BIG_CELEBS_PREFLIGHT_SAMPLES:-64}" \
  --output "${ROOT_DIR}/logs/preflight/${RUN_NAME}.sampling_plan.json"

prepare_comet_record "${ROOT_DIR}" "${RUN_NAME}" "${EXPERIMENT_SPEC_PATH}"

export CONFIG_NAME=big_celebs_scheduled_rhca_40k_bs4
export TRAIN_EPOCH_LEN="${TRAIN_EPOCH_LEN:-2000}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-20}"
export COMET_PROJECT="${COMET_PROJECT:-jul-comet-large-testing-tr}"

exec bash "${SCRIPT_DIR}/run_rhca_apr2026_one_id_1gpu.sh" \
  "trainer.epoch_len=${TRAIN_EPOCH_LEN}" \
  "trainer.n_epochs=${TRAIN_EPOCHS}" \
  "trainer.validation_interval_steps=2000" \
  "trainer.save_period=1" \
  "weights_only_save_period=1" \
  "pipeline.pose_adapt_ratio=0.0" \
  "pipeline.ca_mixing_for_face=false" \
  "$@"
