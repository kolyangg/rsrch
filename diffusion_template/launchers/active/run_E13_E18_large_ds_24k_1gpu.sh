#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/launchers/lib/prepare_comet_record.sh"

: "${RUN_NAME:?Set the unique E13-E18 run name}"
: "${CONFIG_NAME:?Set the matching E13-E18 config name}"
: "${EXPERIMENT_SPEC_PATH:?Set the matching experiment JSON path}"
: "${LARGE_DATASET_MANIFEST:?Set the adjusted identity manifest path}"
: "${LARGE_DATASET_IMAGES:?Set the adjusted image root}"
: "${COMET_API_KEY:?Load COMET_API_KEY from diffusion_template/.env}"

if [[ "$#" -ne 0 ]]; then
  echo "E13-E18 launchers do not accept ad-hoc Hydra overrides." >&2
  exit 2
fi
case "${CONFIG_NAME}" in
  E13_large_ds_joint_shadow_sa128_24k|\
  E14_large_ds_joint_shadow_sa128_protected_24k|\
  E15_large_ds_joint_persist_sa128_protected_24k|\
  E16_large_ds_joint_persist_sa128_idloss_24k|\
  E17_large_ds_joint_persist_sa128_resididca_24k|\
  E18_large_ds_joint_persist_sa128_multiref_24k)
    ;;
  *)
    echo "Unapproved E13-E18 CONFIG_NAME: ${CONFIG_NAME}" >&2
    exit 2
    ;;
esac

python tools/validate_E13_E18_config.py \
  --config-name "${CONFIG_NAME}" \
  --run-name "${RUN_NAME}" \
  --experiment-spec "${EXPERIMENT_SPEC_PATH}"

mkdir -p "${ROOT_DIR}/logs/preflight"
python tools/datasets/preflight_large_dataset.py \
  --manifest "${LARGE_DATASET_MANIFEST}" \
  --images-root "${LARGE_DATASET_IMAGES}" \
  --sample-count "${LARGE_DATASET_PREFLIGHT_SAMPLES:-64}" \
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

exec accelerate launch \
  --config_file=src/configs/ddp/accelerate.yaml \
  --num_processes=1 \
  train.py \
  "--config-name=${CONFIG_NAME}" \
  writer=cometml \
  "writer.run_name=${RUN_NAME}" \
  writer.project_name=aug-large-ds \
  "${MODEL_OVERRIDES[@]}"
