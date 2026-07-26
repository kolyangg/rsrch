#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/launchers/lib/prepare_comet_record.sh"

: "${RUN_NAME:?Set RUN_NAME to the unique evaluation run name}"
: "${VALIDATION_SOURCE_RUN:?Set VALIDATION_SOURCE_RUN to the completed training run}"
: "${VALIDATION_SOURCE_COMET_KEY:?Set the immutable source Comet key}"
: "${FULL96_BBOX_MANUAL:?Set the sealed full-96 manual bbox path}"
: "${EXPERIMENT_SPEC_PATH:?Set the evaluation experiment JSON path}"

export VALIDATION_CHECKPOINT="${ROOT_DIR}/saved/${VALIDATION_SOURCE_RUN}/checkpoint-epoch8.pth"
FULL96_MULTISTEP="${FULL96_MULTISTEP:-false}"
if [[ "${FULL96_MULTISTEP}" != "false" && "${FULL96_MULTISTEP}" != "true" ]]; then
  echo "FULL96_MULTISTEP must be true or false" >&2
  exit 2
fi
if [[ "${FULL96_MULTISTEP}" == "true" ]]; then
  export VALIDATION_CHECKPOINT_EPOCH2="${ROOT_DIR}/saved/${VALIDATION_SOURCE_RUN}/checkpoint-epoch2.pth"
  export VALIDATION_CHECKPOINT_EPOCH4="${ROOT_DIR}/saved/${VALIDATION_SOURCE_RUN}/checkpoint-epoch4.pth"
  export VALIDATION_CHECKPOINT_EPOCH6="${ROOT_DIR}/saved/${VALIDATION_SOURCE_RUN}/checkpoint-epoch6.pth"
  export VALIDATION_CHECKPOINT_EPOCH8="${VALIDATION_CHECKPOINT}"
  for checkpoint in \
    "${VALIDATION_CHECKPOINT_EPOCH2}" \
    "${VALIDATION_CHECKPOINT_EPOCH4}" \
    "${VALIDATION_CHECKPOINT_EPOCH6}" \
    "${VALIDATION_CHECKPOINT_EPOCH8}"; do
    if [[ ! -s "${checkpoint}" ]]; then
      echo "Required multi-step checkpoint is missing: ${checkpoint}" >&2
      exit 3
    fi
  done
fi
SOURCE_IMAGES="${ROOT_DIR}/saved/${VALIDATION_SOURCE_RUN}/val_images/manual_val/step_4000_batch_0"
AUTO_BBOX="${FULL96_BBOX_MANUAL%.json}_auto.json"
SOURCE_REPRO_BBOX_MANUAL="${FULL96_SOURCE_REPRO_BBOX_MANUAL:-${FULL96_BBOX_MANUAL}}"
DUAL_BBOX_PROTOCOL=false
if [[ "$(realpath "${SOURCE_REPRO_BBOX_MANUAL}")" != "$(realpath "${FULL96_BBOX_MANUAL}")" ]]; then
  DUAL_BBOX_PROTOCOL=true
fi

refresh_bbox_protocol() {
  if [[ -n "${FULL96_HISTORICAL_MANUAL:-}" && -n "${FULL96_AUTO_SEED:-}" ]]; then
    python tools/datasets/prepare_full96_validation_protocol.py \
      --historical-manual "${FULL96_HISTORICAL_MANUAL}" \
      --current-auto-seed "${FULL96_AUTO_SEED}" \
      --output-dir "$(dirname "${FULL96_BBOX_MANUAL}")" \
      "$@"
  fi
}

# Provision only the private protocol copy; source bbox files remain untouched.
refresh_bbox_protocol

auto_bbox_count() {
  python3 - "${AUTO_BBOX}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    automatic = json.load(handle)
print(len(automatic))
PY
}

compare_png_batches() {
  local expected_images="$1"
  local candidate_images="$2"
  diff -u \
    <(
      cd "${expected_images}"
      sha256sum ./*.png | sed 's#  \\./#  #' | sort
    ) \
    <(
      cd "${candidate_images}"
      sha256sum ./*.png | sed 's#  \\./#  #' | sort
    )
}

quarantine_new_auto_cache() {
  local initial_count="$1"
  if (( initial_count < 95 )) && [[ -f "${AUTO_BBOX}" ]]; then
    local quarantine_path="${AUTO_BBOX%.json}.failed_$(date -u +%Y%m%dT%H%M%SZ).json"
    mv "${AUTO_BBOX}" "${quarantine_path}"
    echo "Quarantined failed automatic bbox cache: ${quarantine_path}" >&2
    refresh_bbox_protocol
  fi
}

PREREQUISITE_ARGS=()
if [[ -n "${FULL96_REQUIRE_COMPLETED_EVAL:-}" ]]; then
  PREREQUISITE_ARGS+=(
    --auto-min 95
    --require-completed-eval "${FULL96_REQUIRE_COMPLETED_EVAL}"
  )
fi
python tools/inference/check_full96_eval_prerequisites.py \
  --project-root "${ROOT_DIR}" \
  --source-run "${VALIDATION_SOURCE_RUN}" \
  --source-comet-key "${VALIDATION_SOURCE_COMET_KEY}" \
  --bbox-manual "${FULL96_BBOX_MANUAL}" \
  "${PREREQUISITE_ARGS[@]}"

export VALIDATION_CHECKPOINT_SHA256
VALIDATION_CHECKPOINT_SHA256="$(sha256sum "${VALIDATION_CHECKPOINT}" | awk '{print $1}')"
export CONFIG_NAME="cosmic_large_adapted_full96_eval_rhca"
TRACKED_CONFIG_NAME="${CONFIG_NAME}"
if [[ "${FULL96_MULTISTEP}" == "true" ]]; then
  TRACKED_CONFIG_NAME="cosmic_large_adapted_full96_multistep_eval_rhca"
fi
export TRAIN_EPOCHS="8"
export COMET_PROJECT="${COMET_PROJECT:-rsrch-jul}"

# 26 Jul 2026 - AICODE-NOTE: A source trainer may have used a machine-local
# 12-entry automatic bbox cache while the sealed full-96 protocol intentionally
# uses its own 95-entry cache. In the opt-in dual mode, reproduce the trainer
# endpoint under its original bbox routing first, without modifying either
# cache, then use a fresh canonical preflight as the full-96 comparison source.
SOURCE_REPRO_IMAGES="${SOURCE_IMAGES}"
if [[ "${DUAL_BBOX_PROTOCOL}" == "true" ]]; then
  SOURCE_REPRO_RUN="${RUN_NAME}__source_repro_$(date -u +%Y%m%dT%H%M%SZ)_$$"
  WRITER=console RUN_NAME="${SOURCE_REPRO_RUN}" \
    bash "${SCRIPT_DIR}/run_rhca_apr2026_one_id_1gpu.sh" \
      "trainer.from_pretrained=${VALIDATION_CHECKPOINT}" \
      "validation_source_run_name=${VALIDATION_SOURCE_RUN}" \
      "validation_source_comet_key=${VALIDATION_SOURCE_COMET_KEY}" \
      "validation_checkpoint_sha256=${VALIDATION_CHECKPOINT_SHA256}" \
      "datasets.val.manual_val.bbox_mask_gen=${SOURCE_REPRO_BBOX_MANUAL}" \
      "datasets.val.manual_val.limit=12"
  SOURCE_REPRO_IMAGES="${ROOT_DIR}/saved/${SOURCE_REPRO_RUN}/val_images/manual_val/step_4000_batch_0"
  if [[ "$(find "${SOURCE_REPRO_IMAGES}" -maxdepth 1 -type f -name '*.png' | wc -l)" -ne 12 ]] \
    || ! compare_png_batches "${SOURCE_IMAGES}" "${SOURCE_REPRO_IMAGES}"; then
    echo "Source-protocol preflight did not reproduce the trainer endpoint" >&2
    exit 6
  fi
fi

# 25 Jul 2026 - Build and validate the shared automatic-bbox protocol before
# opening the tracked experiment. This is a real reproduction gate and keeps
# bbox overlays from being mixed with the 96 requested Comet generations.
AUTO_COUNT_BEFORE="$(auto_bbox_count)"
if (( AUTO_COUNT_BEFORE < 95 )); then
  PREFLIGHT_LIMIT=96
else
  PREFLIGHT_LIMIT=12
fi
PREFLIGHT_RUN="${RUN_NAME}__protocol_preflight_$(date -u +%Y%m%dT%H%M%SZ)_$$"
set +e
WRITER=console RUN_NAME="${PREFLIGHT_RUN}" \
  bash "${SCRIPT_DIR}/run_rhca_apr2026_one_id_1gpu.sh" \
    "trainer.from_pretrained=${VALIDATION_CHECKPOINT}" \
    "validation_source_run_name=${VALIDATION_SOURCE_RUN}" \
    "validation_source_comet_key=${VALIDATION_SOURCE_COMET_KEY}" \
    "validation_checkpoint_sha256=${VALIDATION_CHECKPOINT_SHA256}" \
    "datasets.val.manual_val.bbox_mask_gen=${FULL96_BBOX_MANUAL}" \
    "datasets.val.manual_val.limit=${PREFLIGHT_LIMIT}"
PREFLIGHT_STATUS=$?
set -e
if (( PREFLIGHT_STATUS != 0 )); then
  quarantine_new_auto_cache "${AUTO_COUNT_BEFORE}"
  exit "${PREFLIGHT_STATUS}"
fi

PREFLIGHT_IMAGES="${ROOT_DIR}/saved/${PREFLIGHT_RUN}/val_images/manual_val/step_4000_batch_0"
if [[ "$(find "${PREFLIGHT_IMAGES}" -maxdepth 1 -type f -name '*.png' | wc -l)" -ne 12 ]]; then
  echo "Protocol preflight did not create 12 first-batch images" >&2
  quarantine_new_auto_cache "${AUTO_COUNT_BEFORE}"
  exit 6
fi
if [[ "${DUAL_BBOX_PROTOCOL}" != "true" ]] \
  && ! compare_png_batches "${SOURCE_IMAGES}" "${PREFLIGHT_IMAGES}"; then
  echo "Protocol preflight did not reproduce the source endpoint's first 12 images" >&2
  quarantine_new_auto_cache "${AUTO_COUNT_BEFORE}"
  exit 6
fi
if [[ "$(auto_bbox_count)" -ne 95 ]]; then
  echo "Protocol preflight did not seal 95 automatic plus one manual route" >&2
  quarantine_new_auto_cache "${AUTO_COUNT_BEFORE}"
  exit 6
fi
refresh_bbox_protocol --require-complete

prepare_comet_record "${ROOT_DIR}" "${RUN_NAME}" "${EXPERIMENT_SPEC_PATH}"

TRACKED_OVERRIDES=(
  "validation_source_run_name=${VALIDATION_SOURCE_RUN}" \
  "validation_source_comet_key=${VALIDATION_SOURCE_COMET_KEY}" \
  "validation_checkpoint_sha256=${VALIDATION_CHECKPOINT_SHA256}" \
  "datasets.val.manual_val.bbox_mask_gen=${FULL96_BBOX_MANUAL}"
)
if [[ "${FULL96_MULTISTEP}" != "true" ]]; then
  TRACKED_OVERRIDES+=(
    "trainer.from_pretrained=${VALIDATION_CHECKPOINT}"
  )
fi
CONFIG_NAME="${TRACKED_CONFIG_NAME}" \
  bash "${SCRIPT_DIR}/run_rhca_apr2026_one_id_1gpu.sh" \
    "${TRACKED_OVERRIDES[@]}"

RECORD="${ROOT_DIR}/saved/${RUN_NAME}/comet_experiment.json"
python tools/comet/comet_experiment.py show "${RECORD}"

EVAL_ROOT="${ROOT_DIR}/saved/${RUN_NAME}/val_images/manual_val"
EXPECTED_FIRST_BATCH_IMAGES="${SOURCE_IMAGES}"
if [[ "${DUAL_BBOX_PROTOCOL}" == "true" ]]; then
  EXPECTED_FIRST_BATCH_IMAGES="${PREFLIGHT_IMAGES}"
fi

if [[ "${FULL96_MULTISTEP}" == "true" ]]; then
  VALIDATION_STEPS=(0 1000 2000 3000 4000)
  for step in "${VALIDATION_STEPS[@]}"; do
    if [[ "$(find "${EVAL_ROOT}" -mindepth 1 -maxdepth 1 -type d -name "step_${step}_batch_*" | wc -l)" -ne 8 ]]; then
      echo "Step ${step} did not create eight full-96 batches" >&2
      exit 5
    fi
    if [[ "$(find "${EVAL_ROOT}"/step_"${step}"_batch_* -maxdepth 1 -type f -name '*.png' | wc -l)" -ne 96 ]]; then
      echo "Step ${step} did not create 96 PNGs" >&2
      exit 5
    fi
  done
  if ! compare_png_batches \
      "${EXPECTED_FIRST_BATCH_IMAGES}" \
      "${EVAL_ROOT}/step_4000_batch_0"; then
    echo "The step-4000 first batch does not reproduce the expected protocol panel" >&2
    exit 6
  fi

  refresh_bbox_protocol --require-complete

  FINALIZER_EXPORT_ARGS=()
  for step in "${VALIDATION_STEPS[@]}"; do
    COMET_EXPORT_ROOT="${ROOT_DIR}/saved/${RUN_NAME}/comet_step${step}_export"
    COMET_EXPORT_JSON="${COMET_EXPORT_ROOT}/comet_runs_export.json"
    COMET_VERIFIED=false
    for attempt in 1 2 3 4 5; do
      if python tools/comet/comet_experiment.py fetch \
          --record "${RECORD}" \
          --step-number "${step}" \
          --output-dir "${COMET_EXPORT_ROOT}"; then
        COMET_VERIFIED=true
        break
      fi
      if (( attempt < 5 )); then
        echo "Comet step-${step} export is incomplete; retrying in 30 seconds" >&2
        sleep 30
      fi
    done
    if [[ "${COMET_VERIFIED}" != "true" ]]; then
      echo "Comet did not expose the step-${step} validation outputs" >&2
      exit 7
    fi
    FINALIZER_EXPORT_ARGS+=(
      --comet-export "${step}=${COMET_EXPORT_JSON}"
    )
  done

  FINALIZER_CHECKPOINT_ARGS=(
    --checkpoint "1000=${VALIDATION_CHECKPOINT_EPOCH2}"
    --checkpoint "2000=${VALIDATION_CHECKPOINT_EPOCH4}"
    --checkpoint "3000=${VALIDATION_CHECKPOINT_EPOCH6}"
    --checkpoint "4000=${VALIDATION_CHECKPOINT_EPOCH8}"
  )
  python tools/inference/finalize_multistep_full96_eval_record.py \
    --record "${RECORD}" \
    --bbox-manual "${FULL96_BBOX_MANUAL}" \
    --images-root "${EVAL_ROOT}" \
    --source-step4000-images "${EXPECTED_FIRST_BATCH_IMAGES}" \
    "${FINALIZER_CHECKPOINT_ARGS[@]}" \
    "${FINALIZER_EXPORT_ARGS[@]}" \
    --verify-only
  python tools/inference/finalize_multistep_full96_eval_record.py \
    --record "${RECORD}" \
    --bbox-manual "${FULL96_BBOX_MANUAL}" \
    --images-root "${EVAL_ROOT}" \
    --source-step4000-images "${EXPECTED_FIRST_BATCH_IMAGES}" \
    "${FINALIZER_CHECKPOINT_ARGS[@]}" \
    "${FINALIZER_EXPORT_ARGS[@]}"

  refresh_bbox_protocol --require-complete
  printf 'FULL96_MULTISTEP_EVAL_COMPLETE run=%s steps=0,1000,2000,3000,4000 images_per_step=96\n' \
    "${RUN_NAME}"
  exit 0
fi

if [[ "$(find "${EVAL_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'step_4000_batch_*' | wc -l)" -ne 8 ]]; then
  echo "Full-96 evaluation did not create eight batches" >&2
  exit 5
fi
if [[ "$(find "${EVAL_ROOT}"/step_4000_batch_* -maxdepth 1 -type f -name '*.png' | wc -l)" -ne 96 ]]; then
  echo "Full-96 evaluation did not create 96 PNGs" >&2
  exit 5
fi

if ! compare_png_batches \
    "${EXPECTED_FIRST_BATCH_IMAGES}" \
    "${EVAL_ROOT}/step_4000_batch_0"; then
  echo "The first 12 full-96 images do not reproduce the expected protocol panel" >&2
  exit 6
fi

refresh_bbox_protocol --require-complete

COMET_EXPORT_ROOT="${ROOT_DIR}/saved/${RUN_NAME}/comet_step4000_export"
COMET_EXPORT_JSON="${COMET_EXPORT_ROOT}/comet_runs_export.json"
FINALIZER_EXTRA_ARGS=()
if [[ "${DUAL_BBOX_PROTOCOL}" == "true" ]]; then
  FINALIZER_EXTRA_ARGS+=(
    --trainer-source-images "${SOURCE_IMAGES}"
    --trainer-reproduction-images "${SOURCE_REPRO_IMAGES}"
    --first-batch-source-kind canonical_protocol_preflight
  )
fi
COMET_VERIFIED=false
for attempt in 1 2 3 4 5; do
  if python tools/comet/comet_experiment.py fetch \
      --record "${RECORD}" \
      --step-number 4000 \
      --output-dir "${COMET_EXPORT_ROOT}" \
    && python tools/inference/finalize_full96_eval_record.py \
      --record "${RECORD}" \
      --checkpoint "${VALIDATION_CHECKPOINT}" \
      --bbox-manual "${FULL96_BBOX_MANUAL}" \
      --images-root "${EVAL_ROOT}" \
      --source-images "${EXPECTED_FIRST_BATCH_IMAGES}" \
      --comet-export "${COMET_EXPORT_JSON}" \
      "${FINALIZER_EXTRA_ARGS[@]}" \
      --verify-only; then
    COMET_VERIFIED=true
    break
  fi
  if (( attempt < 5 )); then
    echo "Comet step-4000 export is not complete yet; retrying in 30 seconds" >&2
    sleep 30
  fi
done
if [[ "${COMET_VERIFIED}" != "true" ]]; then
  echo "Comet did not expose the exact 96 images and both metrics at step 4000" >&2
  exit 7
fi

python tools/inference/finalize_full96_eval_record.py \
  --record "${RECORD}" \
  --checkpoint "${VALIDATION_CHECKPOINT}" \
  --bbox-manual "${FULL96_BBOX_MANUAL}" \
  --images-root "${EVAL_ROOT}" \
  --source-images "${EXPECTED_FIRST_BATCH_IMAGES}" \
  --comet-export "${COMET_EXPORT_JSON}" \
  "${FINALIZER_EXTRA_ARGS[@]}"

refresh_bbox_protocol --require-complete

printf 'FULL96_EVAL_COMPLETE run=%s checkpoint_sha256=%s bbox_auto_sha256=%s images=96\n' \
  "${RUN_NAME}" \
  "${VALIDATION_CHECKPOINT_SHA256}" \
  "$(sha256sum "${AUTO_BBOX}" | awk '{print $1}')"
