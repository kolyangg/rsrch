#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

: "${RUN_NAME:?Set the unique training run name}"
: "${EVAL_RUN_NAME:?Set the unique full-96 evaluation run name}"
: "${EXPERIMENT_SPEC_PATH:?Set the training experiment JSON path}"
: "${EVAL_EXPERIMENT_SPEC_PATH:?Set the evaluation experiment JSON path}"
: "${FULL96_BBOX_MANUAL:?Set the sealed full-96 bbox protocol path}"

TRAIN_RUN_NAME="${RUN_NAME}"

bash "${SCRIPT_DIR}/run_rhca_cosmic_large_initial_usage_1gpu.sh"

TRAIN_RECORD="${ROOT_DIR}/saved/${TRAIN_RUN_NAME}/comet_experiment.json"
if [[ ! -s "${TRAIN_RECORD}" ]]; then
  echo "Training completed without its immutable Comet record: ${TRAIN_RECORD}" >&2
  exit 8
fi
TRAIN_COMET_KEY="$(
  python3 - "${TRAIN_RECORD}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    record = json.load(handle)
key = (record.get("comet") or {}).get("experiment_key")
if not key:
    raise SystemExit("training Comet key is missing")
print(key)
PY
)"
python tools/comet/comet_experiment.py show "${TRAIN_RECORD}"

for epoch in 2 4 6 8; do
  checkpoint="${ROOT_DIR}/saved/${TRAIN_RUN_NAME}/checkpoint-epoch${epoch}.pth"
  if [[ ! -s "${checkpoint}" ]]; then
    echo "Missing checkpoint required for full-96 step $((epoch * 500)): ${checkpoint}" >&2
    exit 9
  fi
done

# The canonical validator opens a second immutable Comet experiment after the
# 4k trainer exits and evaluates 0/1k/2k/3k/4k on the same sealed 96 images.
export RUN_NAME="${EVAL_RUN_NAME}"
export VALIDATION_SOURCE_RUN="${TRAIN_RUN_NAME}"
export VALIDATION_SOURCE_COMET_KEY="${TRAIN_COMET_KEY}"
export FULL96_MULTISTEP=true
export EXPERIMENT_SPEC_PATH="${EVAL_EXPERIMENT_SPEC_PATH}"

bash "${SCRIPT_DIR}/run_rhca_cosmic_full96_eval_1gpu.sh"

printf 'COSMIC_DATASET_USAGE_JOB_COMPLETE train=%s eval=%s\n' \
  "${TRAIN_RUN_NAME}" "${EVAL_RUN_NAME}"
