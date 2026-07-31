#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
ROOT_DIR="$(cd "${ROOT_DIR}" && pwd)"
BASE_LAUNCHER="${ROOT_DIR}/launchers/active/run_rhca_apr2026_one_id_1gpu.sh"
cd "${ROOT_DIR}"

: "${SOURCE_RUN:?Set the existing 4k training run name}"
: "${SOURCE_TRAIN_COMET_KEY:?Set the existing training Comet key}"
: "${EVAL_RUN:?Set the completed 0/1k/2k/3k/4k full-96 run name}"
: "${EVAL_COMET_KEY:?Set the existing full-96 Comet key}"
: "${DATASET_POLICY_ARM:?Set the registered dataset-policy arm}"
: "${CONTINUATION_SPEC_PATH:?Set the continuation experiment JSON path}"
: "${COSMIC_INITIAL_IMAGES_ROOT:?Set the historical Cosmic Large image root}"
: "${FULL96_BBOX_MANUAL:?Set the sealed full-96 manual bbox path}"

CONTINUATION_FIRST_ENDPOINT_EPOCH="${CONTINUATION_FIRST_ENDPOINT_EPOCH:-12}"
case "${CONTINUATION_FIRST_ENDPOINT_EPOCH}" in
  12|16|20|24|28|32|36|40) ;;
  *)
    echo "CONTINUATION_FIRST_ENDPOINT_EPOCH must be 12,16,...,40" >&2
    exit 2
    ;;
esac

COSMIC_INITIAL_METADATA="${COSMIC_INITIAL_METADATA:-${ROOT_DIR}/../dataset_full/cosmic_large_alldata.json}"
COSMIC_INITIAL_CAPTIONS="${COSMIC_INITIAL_CAPTIONS:-${ROOT_DIR}/../dataset_full/all_texts_cosmic_large_by_qween3_trigger_word.json}"
TOPK_TEMPERATURE="0.05"
case "${DATASET_POLICY_ARM}" in
  baseline_self)
    REFERENCE_MODE="self"
    MIN_FACE_RES="0"
    ;;
  distinct_uniform)
    REFERENCE_MODE="uniform"
    MIN_FACE_RES="0"
    ;;
  distinct_highest)
    REFERENCE_MODE="highest_score"
    MIN_FACE_RES="0"
    ;;
  distinct_top3_softmax)
    REFERENCE_MODE="top3_softmax"
    MIN_FACE_RES="0"
    ;;
  target_min256)
    REFERENCE_MODE="self"
    MIN_FACE_RES="256"
    ;;
  *)
    echo "Unknown DATASET_POLICY_ARM=${DATASET_POLICY_ARM}" >&2
    exit 2
    ;;
esac

if [[ "${REFERENCE_MODE}" != "self" ]]; then
  : "${COSMIC_LARGE_MANIFEST:?Distinct-reference arms require the candidate manifest}"
fi

SOURCE_DIR="${ROOT_DIR}/saved/${SOURCE_RUN}"
BASE_EVAL_DIR="${ROOT_DIR}/saved/${EVAL_RUN}"
BASE_EVAL_RECORD="${BASE_EVAL_DIR}/comet_experiment.json"
CONTINUATION_DIR="${ROOT_DIR}/saved_continuations/${EVAL_RUN}"
CONTINUATION_RECORD="${CONTINUATION_DIR}/full96_continuation_record.json"
CONTINUATION_IMAGES="${CONTINUATION_DIR}/val_images/manual_val"
SOURCE_CONFIG_BACKUP="${SOURCE_DIR}/config_before_20k_continuation.yaml"

python3 - \
  "${SOURCE_DIR}/comet_experiment.json" "${SOURCE_TRAIN_COMET_KEY}" \
  "${BASE_EVAL_RECORD}" "${EVAL_COMET_KEY}" \
  "${SOURCE_DIR}/checkpoint-epoch8.pth" <<'PY'
import json
import sys
from pathlib import Path

source_record, source_key, eval_record, eval_key, checkpoint = sys.argv[1:]
for path, expected_key, label in (
    (source_record, source_key, "training"),
    (eval_record, eval_key, "validation"),
):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    found = str((value.get("comet") or {}).get("experiment_key", ""))
    if found != expected_key:
        raise SystemExit(f"{label} Comet key mismatch: {found} != {expected_key}")
    if label == "validation":
        result = value.get("validation_result") or {}
        if result.get("optimizer_steps") != [0, 1000, 2000, 3000, 4000]:
            raise SystemExit("base full-96 validation is not complete")
checkpoint_path = Path(checkpoint)
if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
    raise SystemExit(f"missing 4k checkpoint: {checkpoint_path}")
print("CONTINUATION_SOURCE_RECORDS_VERIFIED")
PY

python tools/inference/check_full96_eval_prerequisites.py \
  --project-root "${ROOT_DIR}" \
  --source-run "${SOURCE_RUN}" \
  --source-comet-key "${SOURCE_TRAIN_COMET_KEY}" \
  --bbox-manual "${FULL96_BBOX_MANUAL}" \
  --auto-min 95 \
  --require-completed-eval "${EVAL_RUN}"

PREFLIGHT_ARGS=(
  --metadata "${COSMIC_INITIAL_METADATA}"
  --captions "${COSMIC_INITIAL_CAPTIONS}"
  --images-root "${COSMIC_INITIAL_IMAGES_ROOT}"
  --reference-mode "${REFERENCE_MODE}"
  --min-face-res "${MIN_FACE_RES}"
  --topk-temperature "${TOPK_TEMPERATURE}"
  --sample-count "${COSMIC_PREFLIGHT_SAMPLES:-64}"
  --output "${ROOT_DIR}/logs/preflight/${SOURCE_RUN}_continue20k.json"
)
if [[ "${REFERENCE_MODE}" != "self" ]]; then
  PREFLIGHT_ARGS+=(--candidate-manifest "${COSMIC_LARGE_MANIFEST}")
fi
mkdir -p "${ROOT_DIR}/logs/preflight" "${CONTINUATION_DIR}"
python tools/datasets/preflight_cosmic_large_initial_usage.py "${PREFLIGHT_ARGS[@]}"

if [[ ! -f "${SOURCE_CONFIG_BACKUP}" ]]; then
  cp --preserve=mode,timestamps "${SOURCE_DIR}/config.yaml" "${SOURCE_CONFIG_BACKUP}"
fi
if [[ -f "${CONTINUATION_DIR}/experiment_plan.json" ]]; then
  cmp --silent "${CONTINUATION_SPEC_PATH}" "${CONTINUATION_DIR}/experiment_plan.json"
else
  cp --preserve=mode,timestamps \
    "${CONTINUATION_SPEC_PATH}" \
    "${CONTINUATION_DIR}/experiment_plan.json"
fi

if (( CONTINUATION_FIRST_ENDPOINT_EPOCH > 12 )); then
  resume_step=$(((CONTINUATION_FIRST_ENDPOINT_EPOCH - 4) * 500))
  python3 - "${CONTINUATION_RECORD}" "${resume_step}" <<'PY'
import json
import sys
from pathlib import Path

record_path, resume_step = Path(sys.argv[1]), int(sys.argv[2])
if not record_path.is_file():
    raise SystemExit(f"missing continuation record: {record_path}")
record = json.loads(record_path.read_text(encoding="utf-8"))
expected = list(range(6000, resume_step + 1, 2000))
if record.get("completed_optimizer_steps") != expected:
    raise SystemExit(
        "continuation recovery prefix mismatch: "
        f"{record.get('completed_optimizer_steps')} != {expected}"
    )
print(f"CONTINUATION_RECOVERY_PREFIX_VERIFIED steps={expected}")
PY
fi

# 27 Jul 2026 - AICODE-NOTE: epochs remain 500 optimizer steps so checkpoint
# epoch numbers retain their original meaning. Four validation-free epochs are
# resumed at a time, then the exact sealed 96-image evaluator appends that
# 2,000-step endpoint to the pre-existing validation Comet experiment.
for endpoint_epoch in 12 16 20 24 28 32 36 40; do
  if (( endpoint_epoch < CONTINUATION_FIRST_ENDPOINT_EPOCH )); then
    continue
  fi
  resume_epoch=$((endpoint_epoch - 4))
  endpoint_step=$((endpoint_epoch * 500))
  resume_checkpoint_name="checkpoint-epoch${resume_epoch}.pth"
  endpoint_checkpoint="${SOURCE_DIR}/checkpoint-epoch${endpoint_epoch}.pth"

  if [[ -e "${endpoint_checkpoint}" ]]; then
    echo "Refusing to reuse an existing continuation checkpoint: ${endpoint_checkpoint}" >&2
    exit 10
  fi

  CONFIG_NAME="cosmic_large_initial_usage_rhca" \
  WRITER="cometml" \
  RUN_NAME="${SOURCE_RUN}" \
  TRAIN_EPOCHS="${endpoint_epoch}" \
  bash "${BASE_LAUNCHER}" \
    "cometml_id=${SOURCE_TRAIN_COMET_KEY}" \
    "continue_run=true" \
    "saved_checkpoint=${resume_checkpoint_name}" \
    "trainer.resume_from=${resume_checkpoint_name}" \
    "trainer.save_period=4" \
    "weights_only_save_period=0" \
    "val_datasets_names=[]" \
    "dataset_policy.reference_mode=${REFERENCE_MODE}" \
    "dataset_policy.min_face_res=${MIN_FACE_RES}" \
    "dataset_policy.topk_temperature=${TOPK_TEMPERATURE}" \
    "pipeline.pose_adapt_ratio=0.0" \
    "pipeline.ca_mixing_for_face=false"

  if [[ ! -s "${endpoint_checkpoint}" ]]; then
    echo "Training segment did not create ${endpoint_checkpoint}" >&2
    exit 11
  fi

  validation_checkpoint_sha256="$(sha256sum "${endpoint_checkpoint}" | awk '{print $1}')"
  export VALIDATION_SOURCE_RUN="${SOURCE_RUN}"
  export VALIDATION_SOURCE_COMET_KEY="${SOURCE_TRAIN_COMET_KEY}"
  export VALIDATION_CHECKPOINT="${endpoint_checkpoint}"
  export VALIDATION_CHECKPOINT_SHA256="${validation_checkpoint_sha256}"

  CONFIG_NAME="cosmic_large_adapted_full96_eval_rhca" \
  WRITER="cometml" \
  RUN_NAME="${EVAL_RUN}" \
  TRAIN_EPOCHS="${endpoint_epoch}" \
  bash "${BASE_LAUNCHER}" \
    "cometml_id=${EVAL_COMET_KEY}" \
    "trainer.from_pretrained=${endpoint_checkpoint}" \
    "trainer.save_dir=saved_continuations" \
    "validation_epoch=${endpoint_epoch}" \
    "datasets.val.manual_val.bbox_mask_gen=${FULL96_BBOX_MANUAL}" \
    "datasets.val.manual_val.limit=96" \
    "pipeline.pose_adapt_ratio=0.0" \
    "pipeline.ca_mixing_for_face=false"

  if [[ "$(find "${CONTINUATION_IMAGES}" -mindepth 1 -maxdepth 1 -type d -name "step_${endpoint_step}_batch_*" | wc -l)" -ne 8 ]]; then
    echo "Step ${endpoint_step} did not create eight full-96 batches" >&2
    exit 12
  fi
  if [[ "$(find "${CONTINUATION_IMAGES}"/step_"${endpoint_step}"_batch_* -maxdepth 1 -type f -name '*.png' | wc -l)" -ne 96 ]]; then
    echo "Step ${endpoint_step} did not create 96 PNGs" >&2
    exit 12
  fi

  comet_export_root="${CONTINUATION_DIR}/comet_step${endpoint_step}_export"
  comet_export_json="${comet_export_root}/comet_runs_export.json"
  comet_verified=false
  # 27 Jul 2026 - A successful HTTP export can still contain a transiently
  # truncated asset or metrics that have not finished propagating. Treat the
  # pixel/metric verifier as part of the retryable Comet transaction.
  for attempt in 1 2 3 4 5; do
    if python tools/comet/comet_experiment.py fetch \
        --record "${BASE_EVAL_RECORD}" \
        --step-number "${endpoint_step}" \
        --output-dir "${comet_export_root}" \
        && python tools/inference/verify_appended_full96_step.py \
          --base-record "${BASE_EVAL_RECORD}" \
          --continuation-record "${CONTINUATION_RECORD}" \
          --bbox-manual "${FULL96_BBOX_MANUAL}" \
          --images-root "${CONTINUATION_IMAGES}" \
          --checkpoint "${endpoint_checkpoint}" \
          --comet-export "${comet_export_json}" \
          --step "${endpoint_step}"; then
      comet_verified=true
      break
    fi
    if (( attempt < 5 )); then
      echo "Comet step-${endpoint_step} export is incomplete; retrying in 30 seconds" >&2
      sleep 30
    fi
  done
  if [[ "${comet_verified}" != "true" ]]; then
    echo "Comet did not expose the appended step-${endpoint_step} outputs" >&2
    exit 13
  fi

  printf 'CONTINUATION_GATE_COMPLETE source=%s step=%s images=96 comet=%s\n' \
    "${SOURCE_RUN}" "${endpoint_step}" "${EVAL_COMET_KEY}"
done

printf 'CONTINUATION_20K_COMPLETE source=%s eval=%s steps=6000,8000,10000,12000,14000,16000,18000,20000\n' \
  "${SOURCE_RUN}" "${EVAL_RUN}"
