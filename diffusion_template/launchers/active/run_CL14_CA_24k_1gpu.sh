#!/usr/bin/env bash
# CL14_CA: CL14 plus bounded residual PhotoMaker-ID cross-attention.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/launchers/lib/prepare_comet_record.sh"

: "${RUN_NAME:?Set the unique CL14_CA run name}"
: "${CONFIG_NAME:?Set CL14_CA or CL14_CA_skipval_smoke}"
: "${EXPERIMENT_SPEC_PATH:?Set the CL14_CA experiment JSON path}"
: "${COSMIC_LARGE_MANIFEST:?Set the filtered Cosmic manifest}"
: "${COSMIC_LARGE_ROOT:?Set the Cosmic image root}"
: "${COMET_API_KEY:?Load COMET_API_KEY from diffusion_template/.env}"
: "${FACE_QUALITY_SCORER_PYTHON:?Set the PyIQA scorer interpreter}"

if [[ "$#" -ne 0 ]]; then
  echo "CL14_CA rejects ad-hoc Hydra overrides." >&2
  exit 2
fi
case "${RUN_NAME}:${CONFIG_NAME}" in
  # 12 Aug 2026 - Training optimization runs preserve the CL14_CA science.
  CL14_CA:CL14_CA|CL14_CA_r3:CL14_CA|CL14_CA_r4:CL14_CA|CL14_CA_r5:CL14_CA|CL14_CA_r6:CL14_CA|CL14_CA_r7:CL14_CA|CL14_CA_optimized_r1:CL14_CA|CL14_CA_optimized_speed_smoke_r1:CL14_CA_skipval_smoke|CL14_CA_skipval_smoke_r1:CL14_CA_skipval_smoke|CL14_CA_skipval_smoke_r2:CL14_CA_skipval_smoke|CL14_CA_skipval_smoke_r3:CL14_CA_skipval_smoke|CL14_CA_skipval_smoke_r4:CL14_CA_skipval_smoke|CL14_CA_skipval_smoke_r5:CL14_CA_skipval_smoke|CL14_CA_oneval_smoke_r1:CL14_CA_oneval_smoke|CL14_CA_onebatch_smoke_r1:CL14_CA_onebatch_smoke|CL14_CA_onebatch_smoke_r2:CL14_CA_onebatch_smoke) ;;
  *) echo "Unexpected CL14_CA run/config pair" >&2; exit 2 ;;
esac

test -s "${COSMIC_LARGE_MANIFEST}"
test -d "${COSMIC_LARGE_ROOT}"
VAL_ROOT="${ROOT_DIR}/../dataset_full/val_dataset"
for sealed_file in \
  "e8fb3290e6da6eacc70c6cc67f2affa0c923c1ca605efc35ddca95ee48f1ebaf prompts_10.txt" \
  "d1f53322d6964c2d30d28ef2cc765366a42776117e3982909d6fdfc1ae99872b classes_ref.json" \
  "eadb9411b9d0b98238714bb263db708e56a30abee91c67c4df0c7e1e5c4a268f ref_bboxes.json" \
  "dd3b2c1ea5eebd7fcd52128b5b7b36a8623996b6601dcd5362adc26f65ed9c7d pm96_bboxes_new.json"; do
  read -r expected_sha relative_path <<<"${sealed_file}"
  test "$(sha256sum "${VAL_ROOT}/${relative_path}" | cut -d' ' -f1)" = "${expected_sha}"
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

python tools/validate_CL14_CA_config.py \
  --config-name "${CONFIG_NAME}" \
  --run-name "${RUN_NAME}" \
  --experiment-spec "${EXPERIMENT_SPEC_PATH}"
mkdir -p "${ROOT_DIR}/logs/preflight"
python tools/datasets/preflight_cosmic_cl.py \
  --config-name "${CONFIG_NAME}" \
  --sample-count "${COSMIC_PREFLIGHT_SAMPLES:-64}" \
  --output "${ROOT_DIR}/logs/preflight/${RUN_NAME}.json"

if [[ "${CL14_CA_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "CL14_CA exact launcher preflight complete: ${RUN_NAME}"
  exit 0
fi

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
  echo "Training failed with status ${TRAIN_STATUS}; deferred face quality skipped." >&2
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
