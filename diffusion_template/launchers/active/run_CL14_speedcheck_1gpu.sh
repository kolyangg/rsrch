#!/usr/bin/env bash
# Bounded operational CL14 throughput qualification on the current pipeline.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/launchers/lib/prepare_comet_record.sh"

: "${RUN_NAME:?Set the unique CL14 speedcheck run name}"
: "${EXPERIMENT_SPEC_PATH:?Set the experiment JSON path}"
: "${COSMIC_LARGE_MANIFEST:?Set the filtered Cosmic manifest}"
: "${COSMIC_LARGE_ROOT:?Set the Cosmic image root}"
: "${COMET_API_KEY:?Load COMET_API_KEY from .env}"
: "${FACE_QUALITY_SCORER_PYTHON:?Set the PyIQA scorer interpreter}"
: "${SUBJECT_V2_ID_EMBEDS:?Set sealed subject-v2 embeddings}"
if [[ "$#" -ne 0 ]]; then echo "Ad-hoc overrides are rejected." >&2; exit 2; fi

CONFIG_NAME="CL14_cosmic_joint_shadow_sa128_softmask_24k_speedcheck"
python tools/validate_CL14_speedcheck_config.py \
  --run-name "${RUN_NAME}" --experiment-spec "${EXPERIMENT_SPEC_PATH}"
mkdir -p "${ROOT_DIR}/logs/preflight"
python tools/datasets/preflight_cosmic_cl.py \
  --config-name "${CONFIG_NAME}" --sample-count "${COSMIC_PREFLIGHT_SAMPLES:-64}" \
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
MODEL_OVERRIDES+=("metrics.id_sim_subject_v2.id_embeds_pth=${SUBJECT_V2_ID_EMBEDS}")
exec accelerate launch \
  --config_file=src/configs/ddp/accelerate.yaml \
  --num_processes=1 \
  train.py \
  "--config-name=${CONFIG_NAME}" \
  writer=cometml \
  "writer.run_name=${RUN_NAME}" \
  writer.project_name=aug-large-ds \
  "${MODEL_OVERRIDES[@]}"
