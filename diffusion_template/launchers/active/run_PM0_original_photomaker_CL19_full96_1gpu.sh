#!/usr/bin/env bash
# PM0: original PhotoMaker V2 on the exact CL19 fixed-96 validation contract.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${PM_BASELINE_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/launchers/lib/prepare_comet_record.sh"

: "${RUN_NAME:?Set the unique baseline run name}"
: "${EXPERIMENT_SPEC_PATH:?Set the baseline experiment JSON path}"
: "${COSMIC_LARGE_MANIFEST:?Set the CL19 Cosmic manifest}"
: "${COSMIC_LARGE_ROOT:?Set the CL19 Cosmic image root}"
: "${COMET_API_KEY:?Load COMET_API_KEY from the machine-local .env}"
: "${FACE_QUALITY_SCORER_PYTHON:?Set the PyIQA 0.1.15 interpreter}"
: "${SUBJECT_V2_ID_EMBEDS:?Set the sealed subject-v2 identity embeddings}"
: "${PM_PATH:?Set the PhotoMaker V2 checkpoint path}"

if [[ "$#" -ne 0 ]]; then
  echo "PM0 rejects ad-hoc Hydra overrides." >&2
  exit 2
fi

CONFIG_NAME="PM0_original_photomaker_CL19_full96"
BASE_CONFIG_NAME="CL19_cosmic_true_soft_fullquery_router_24k"
VAL_ROOT="${ROOT_DIR}/../dataset_full/val_dataset"

test -s "${PM_PATH}"
test -s "${COSMIC_LARGE_MANIFEST}"
test -d "${COSMIC_LARGE_ROOT}"
test -s "${SUBJECT_V2_ID_EMBEDS}"
test -s "${EXPERIMENT_SPEC_PATH}"
test -f "${ROOT_DIR}/src/configs/${CONFIG_NAME}.yaml"

for sealed_file in \
  "e8fb3290e6da6eacc70c6cc67f2affa0c923c1ca605efc35ddca95ee48f1ebaf prompts_10.txt" \
  "d1f53322d6964c2d30d28ef2cc765366a42776117e3982909d6fdfc1ae99872b classes_ref.json" \
  "eadb9411b9d0b98238714bb263db708e56a30abee91c67c4df0c7e1e5c4a268f ref_bboxes.json" \
  "dd3b2c1ea5eebd7fcd52128b5b7b36a8623996b6601dcd5362adc26f65ed9c7d pm96_bboxes_new.json"; do
  read -r expected_sha relative_path <<<"${sealed_file}"
  actual_sha="$(sha256sum "${VAL_ROOT}/${relative_path}" | cut -d' ' -f1)"
  test "${actual_sha}" = "${expected_sha}"
done

reference_sha="$({
  find "${VAL_ROOT}/references" -type f -printf '%P\n' | LC_ALL=C sort |
    while read -r relative_path; do
      printf '%s  %s\n' \
        "$(sha256sum "${VAL_ROOT}/references/${relative_path}" | cut -d' ' -f1)" \
        "${relative_path}"
    done
} | sha256sum | cut -d' ' -f1)"
test "${reference_sha}" = "7297fe241273914ec2d401952bea0c83730beb5a58ebf3820b0bf50dac22606e"
test "$(sha256sum "${SUBJECT_V2_ID_EMBEDS}" | cut -d' ' -f1)" = \
  "e0d36212ad350db8252c4805acf46aa4c90289603d460584dc7692066712b465"

# Fail closed if the dedicated config drifts from CL19 in anything except the
# four declared validation-only/plain-PhotoMaker fields and its Comet comment.
python - "${ROOT_DIR}/src/configs" "${BASE_CONFIG_NAME}" "${CONFIG_NAME}" <<'PY'
from __future__ import annotations

import sys
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

config_dir, base_name, baseline_name = sys.argv[1:]
with initialize_config_dir(version_base=None, config_dir=config_dir):
    base = OmegaConf.to_container(compose(config_name=base_name), resolve=True)
    baseline = OmegaConf.to_container(compose(config_name=baseline_name), resolve=True)

missing = object()

def flatten(value, prefix=""):
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten(child, path))
        return result
    return {prefix: value}

base_flat = flatten(base)
baseline_flat = flatten(baseline)
different = {
    key
    for key in set(base_flat) | set(baseline_flat)
    if base_flat.get(key, missing) != baseline_flat.get(key, missing)
}
allowed = {
    "validation_only",
    "validation_epochs",
    "validation_checkpoint_paths",
    "validation_args.use_branched_attention",
    "writer.experiment_comment",
}
if different != allowed:
    raise RuntimeError(
        "PM0/CL19 config delta mismatch: "
        f"unexpected={sorted(different - allowed)}, missing={sorted(allowed - different)}"
    )

assert baseline["validation_only"] is True
assert baseline["validation_epochs"] == [0]
assert baseline["validation_checkpoint_paths"] == [None]
assert baseline["validation_args"]["use_branched_attention"] is False
assert baseline["pretrained_model_for_validation_name_or_path"] == "SG161222/RealVisXL_V4.0"
assert baseline["validation_processor_base_mode"] == "legacy_full_copy"
assert baseline["strict_validation_processor_copy"] is True
assert baseline["validation_shadow_photomaker_default"] is True
assert baseline["trainer"]["seed"] == 0
assert baseline["trainer"]["log_per_image_id_sim_table"] is True
assert baseline["trainer"]["face_quality"]["enabled"] is True
assert baseline["trainer"]["face_quality"]["expected_images"] == 96
assert baseline["trainer"]["face_quality"]["execution_mode"] == "deferred"
assert baseline["datasets"]["val"]["manual_val"]["seeds"] == [0]
assert baseline["datasets"]["val"]["manual_val"]["limit"] == 96
assert baseline["datasets"]["val"]["manual_val"]["face_subject_selection_policy"] == "bbox_overlap_v2"
assert baseline["dataloaders"]["manual_val"]["batch_size"] == 12
assert baseline["validation_args"]["num_images_per_prompt"] == 1
assert baseline["validation_args"]["num_inference_steps"] == 50
assert baseline["validation_args"]["guidance_scale"] == 5
assert baseline["validation_args"]["photomaker_start_step"] == 10
assert baseline["validation_args"]["photomaker_use_lora_adapter"] is False
assert baseline["pipeline"]["pose_adapt_ratio"] == 0.0
assert baseline["pipeline"]["ca_mixing_for_face"] is False
assert baseline["inference_metrics"] == [
    "clip_ts",
    "id_sim_best_legacy",
    "id_sim_subject_v2",
]
print("PM0_CL19_CONFIG_GATE_OK exact_delta=5 full96=96 batch=12 seed=0 DDIM_steps=50 CFG=5")
PY

prepare_comet_record "${ROOT_DIR}" "${RUN_NAME}" "${EXPERIMENT_SPEC_PATH}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HYDRA_FULL_ERROR=1
export ACCELERATE_LOG_LEVEL=error
export TRANSFORMERS_VERBOSITY=error
export DIFFUSERS_VERBOSITY=error
export COMET_DISABLE_AUTO_LOGGING=1
export COMET_LOGGING_CONSOLE=ERROR
export ACCELERATE_NUM_PROCESSES=1

set +e
accelerate launch \
  --config_file=src/configs/ddp/accelerate.yaml \
  --num_processes=1 \
  train.py \
  "--config-name=${CONFIG_NAME}" \
  writer=cometml \
  "writer.run_name=${RUN_NAME}" \
  writer.project_name=aug-large-ds \
  "model.photomaker_path=${PM_PATH}" \
  "metrics.id_sim_subject_v2.id_embeds_pth=${SUBJECT_V2_ID_EMBEDS}" &
INFER_PID=$!
set -e

COMET_RECORD="${ROOT_DIR}/saved/${RUN_NAME}/comet_experiment.json"
COMET_READY=0
for _ in $(seq 1 300); do
  if [[ -s "${COMET_RECORD}" ]] && python - "${COMET_RECORD}" <<'PY'
import json, sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
key = (record.get("comet") or {}).get("experiment_key")
raise SystemExit(0 if isinstance(key, str) and len(key) == 32 else 1)
PY
  then
    COMET_READY=1
    echo "COMET_STARTUP_VERIFIED ${COMET_RECORD}"
    break
  fi
  if ! kill -0 "${INFER_PID}" 2>/dev/null; then
    wait "${INFER_PID}"
    exit $?
  fi
  sleep 2
done
if [[ "${COMET_READY}" -ne 1 ]]; then
  echo "Comet immutable key was not registered within 10 minutes." >&2
  kill "${INFER_PID}" 2>/dev/null || true
  wait "${INFER_PID}" || true
  exit 78
fi

set +e
wait "${INFER_PID}"
INFER_STATUS=$?
set -e
if [[ "${INFER_STATUS}" -ne 0 ]]; then
  echo "Original PhotoMaker validation failed with status ${INFER_STATUS}." >&2
  exit "${INFER_STATUS}"
fi

"${FACE_QUALITY_SCORER_PYTHON}" \
  tools/comet/finalize_deferred_face_quality.py \
  --run-dir "${ROOT_DIR}/saved/${RUN_NAME}" \
  --expected-project aug-large-ds \
  --expected-steps 0 \
  --images-per-step 96 \
  --partition manual_val \
  --scorer-python "${FACE_QUALITY_SCORER_PYTHON}" \
  --device cuda \
  --batch-size 8 \
  --write \
  --upload-per-image-asset

python - "${ROOT_DIR}/saved/${RUN_NAME}" <<'PY'
import csv
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
record = json.loads((run_dir / "comet_experiment.json").read_text(encoding="utf-8"))
key = (record.get("comet") or {}).get("experiment_key")
if not isinstance(key, str) or len(key) != 32:
    raise RuntimeError("Missing immutable Comet experiment key")

table = run_dir / "validation_tables" / "id_sim__manual_val__step_000000.csv"
with table.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != 96 or [int(row["image_index"]) for row in rows] != list(range(96)):
    raise RuntimeError("Per-image identity table is not the exact 96-row panel")

status = json.loads(
    (run_dir / "post_training_face_quality" / "status.json").read_text(encoding="utf-8")
)
if not (
    status.get("status") == "complete"
    and status.get("comet_written") is True
    and status.get("per_image_asset_uploaded") is True
    and status.get("steps") == [0]
    and status.get("images_per_step") == 96
):
    raise RuntimeError(f"Face-quality finalization incomplete: {status}")
print(f"PM0_CL19_LOCAL_OUTPUT_GATE_OK comet_key={key} id_rows=96 face_quality_steps=1")
PY
