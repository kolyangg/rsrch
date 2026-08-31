#!/usr/bin/env bash
# One allocation: no-val 100, fixed-96+100, then the gate-approved 24k run.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
: "${RUN_NAME:?Set production run name}"
: "${CONFIG_NAME:?Set production config}"
: "${EXPERIMENT_SPEC_PATH:?Set production experiment record}"
: "${NO_VAL_CONFIG_NAME:?Set sealed no-validation config}"
: "${VALIDATED_CONFIG_NAME:?Set sealed validated config}"
: "${CL39N_ARM:?Set CL39N6R, CL39N7, CL39N8, or CL39N9}"
: "${CL39N_QUAL_ROOT:?Set a fresh absent qualification root}"
if [[ "$#" -ne 0 ]]; then echo "Qualification wrapper rejects overrides" >&2; exit 2; fi
test ! -e "${CL39N_QUAL_ROOT}"
mkdir -p "${CL39N_QUAL_ROOT}/saved" "${CL39N_QUAL_ROOT}/logs" "${CL39N_QUAL_ROOT}/gates"
cp -p "${ROOT_DIR}/../dataset_full/val_dataset/pm96_bboxes_new.json" \
  "${CL39N_QUAL_ROOT}/pm96_bboxes_seed0.json"
export CL39N_BBOX_PATH="${CL39N_QUAL_ROOT}/pm96_bboxes_seed0.json"
export PHOTOMAKER_FACEANALYSIS_CPU=1
CEILING="${CL39N_MAX_MEDIAN_SECONDS:-5.0}"
NO_VAL_RUN="${RUN_NAME}_qual_noval100"
VALIDATED_RUN="${RUN_NAME}_qual_validated100"

# 31 Aug 2026 - Qualifications must consume the same sealed architecture and
# Cosmic records as production; fail before GPU work if either contract drifts.
python tools/validate_CL39N6R_CL39N9_config.py \
  --config-name "${CONFIG_NAME}" --run-name "${RUN_NAME}" \
  --experiment-spec "${EXPERIMENT_SPEC_PATH}"
mkdir -p "${CL39N_QUAL_ROOT}/preflight"
python tools/datasets/preflight_cosmic_cl.py --config-name "${CONFIG_NAME}" \
  --sample-count "${COSMIC_PREFLIGHT_SAMPLES:-64}" \
  --output "${CL39N_QUAL_ROOT}/preflight/${RUN_NAME}.json"

accelerate launch --config_file=src/configs/ddp/accelerate.yaml --num_processes=1 \
  train.py "--config-name=${NO_VAL_CONFIG_NAME}" writer=console \
  "writer.run_name=${NO_VAL_RUN}" "trainer.save_dir=${CL39N_QUAL_ROOT}/saved" \
  2>&1 | tee "${CL39N_QUAL_ROOT}/logs/no_validation.log"
python tools/analysis/check_cl39n_training_smoke.py \
  --log "${CL39N_QUAL_ROOT}/logs/no_validation.log" --arm "${CL39N_ARM}" \
  --mode no_validation --max-median-seconds "${CEILING}" \
  --output "${CL39N_QUAL_ROOT}/gates/no_validation.json"

accelerate launch --config_file=src/configs/ddp/accelerate.yaml --num_processes=1 \
  train.py "--config-name=${VALIDATED_CONFIG_NAME}" writer=console \
  "writer.run_name=${VALIDATED_RUN}" "trainer.save_dir=${CL39N_QUAL_ROOT}/saved" \
  2>&1 | tee "${CL39N_QUAL_ROOT}/logs/validated.log"
VALIDATED_IMAGES="${CL39N_QUAL_ROOT}/saved/${VALIDATED_RUN}/val_images/manual_val"
python tools/analysis/check_cl39n_training_smoke.py \
  --log "${CL39N_QUAL_ROOT}/logs/validated.log" --arm "${CL39N_ARM}" \
  --mode validated --max-median-seconds "${CEILING}" \
  --expected-images 96 --images-root "${VALIDATED_IMAGES}" \
  --output "${CL39N_QUAL_ROOT}/gates/validated.json"

echo "CL39N_QUALIFICATION_PASSED arm=${CL39N_ARM} ceiling=${CEILING}s"
exec bash launchers/active/run_CL39N6R_CL39N9_1gpu.sh
