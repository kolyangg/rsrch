#!/usr/bin/env bash
# CL21-CL26: one-delta CL19 follow-ups on the fixed full96 contract.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/launchers/lib/prepare_comet_record.sh"

: "${RUN_NAME:?Set the unique CL21-CL26 run name}"
: "${CONFIG_NAME:?Set the matching config name}"
: "${EXPERIMENT_SPEC_PATH:?Set the immutable experiment JSON path}"
: "${COSMIC_LARGE_MANIFEST:?Set the filtered Cosmic manifest}"
: "${COSMIC_LARGE_ROOT:?Set the Cosmic image root}"
: "${COMET_API_KEY:?Load COMET_API_KEY from diffusion_template/.env}"
: "${FACE_QUALITY_SCORER_PYTHON:?Set the PyIQA scorer interpreter}"
: "${SUBJECT_V2_ID_EMBEDS:?Set the sealed subject-v2 identity embeddings}"

if [[ "$#" -ne 0 ]]; then
  echo "CL21-CL26 launchers reject ad-hoc Hydra overrides." >&2
  exit 2
fi
case "${CONFIG_NAME}" in
  CL21_cosmic_true_soft_router_resididca_v3_24k|\
  CL22_cosmic_visibility_order_router_24k|\
  CL23_cosmic_temporal_frequency_router_24k|\
  CL24_cosmic_pm_boundary_distill_24k|\
  CL25_cosmic_low_noise_id_reward_4k|\
  CL26_cosmic_anchored_highres_roi_ba_24k) ;;
  *) echo "Unapproved CL21-CL26 config: ${CONFIG_NAME}" >&2; exit 2 ;;
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

if [[ "${CONFIG_NAME}" == CL25_* ]]; then
  : "${CL25_SOURCE_CHECKPOINT:?Set the immutable CL19 24k weights-only checkpoint}"
  : "${CL25_SOURCE_SHA256:?Set the pinned CL19 checkpoint hash}"
  : "${ARCFACE_ONNX_PATH:?Set the frozen buffalo_l recognition graph}"
  test -s "${CL25_SOURCE_CHECKPOINT}"
  test -s "${ARCFACE_ONNX_PATH}"
  test "$(sha256sum "${CL25_SOURCE_CHECKPOINT}" | cut -d' ' -f1)" = "${CL25_SOURCE_SHA256}"
  test "$(sha256sum "${ARCFACE_ONNX_PATH}" | cut -d' ' -f1)" = "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"
fi

python tools/validate_CL21_CL26_config.py \
  --config-name "${CONFIG_NAME}" \
  --run-name "${RUN_NAME}" \
  --experiment-spec "${EXPERIMENT_SPEC_PATH}"
mkdir -p "${ROOT_DIR}/logs/preflight"
python tools/datasets/preflight_cosmic_cl.py \
  --config-name "${CONFIG_NAME}" \
  --sample-count "${COSMIC_PREFLIGHT_SAMPLES:-64}" \
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

MODEL_OVERRIDES=("metrics.id_sim_subject_v2.id_embeds_pth=${SUBJECT_V2_ID_EMBEDS}")
if [[ -n "${PM_PATH:-}" ]]; then MODEL_OVERRIDES+=("model.photomaker_path=${PM_PATH}"); fi

set +e
accelerate launch --config_file=src/configs/ddp/accelerate.yaml --num_processes=1 \
  train.py "--config-name=${CONFIG_NAME}" writer=cometml \
  "writer.run_name=${RUN_NAME}" writer.project_name=aug-large-ds \
  "${MODEL_OVERRIDES[@]}" &
TRAIN_PID=$!
set -e

COMET_RECORD="${ROOT_DIR}/saved/${RUN_NAME}/comet_experiment.json"
COMET_READY=0
for _ in $(seq 1 300); do
  if [[ -s "${COMET_RECORD}" ]] && python - "${COMET_RECORD}" <<'PY'
import json, sys
key = (json.load(open(sys.argv[1], encoding="utf-8")).get("comet") or {}).get("experiment_key")
raise SystemExit(0 if isinstance(key, str) and len(key) == 32 else 1)
PY
  then COMET_READY=1; echo "COMET_STARTUP_VERIFIED ${COMET_RECORD}"; break; fi
  if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then wait "${TRAIN_PID}"; exit $?; fi
  sleep 2
done
if [[ "${COMET_READY}" -ne 1 ]]; then
  echo "Comet immutable key was not registered within 10 minutes." >&2
  kill "${TRAIN_PID}" 2>/dev/null || true; wait "${TRAIN_PID}" || true; exit 78
fi
set +e; wait "${TRAIN_PID}"; TRAIN_STATUS=$?; set -e
if [[ "${TRAIN_STATUS}" -ne 0 ]]; then exit "${TRAIN_STATUS}"; fi

EXPECTED_STEPS="0,2000,4000,6000,8000,10000,12000,14000,16000,18000,20000,22000,24000"
if [[ "${CONFIG_NAME}" == CL25_* ]]; then EXPECTED_STEPS="0,2000,4000"; fi
"${FACE_QUALITY_SCORER_PYTHON}" tools/comet/finalize_deferred_face_quality.py \
  --run-dir "${ROOT_DIR}/saved/${RUN_NAME}" --expected-project aug-large-ds \
  --expected-steps "${EXPECTED_STEPS}" --images-per-step 96 --partition manual_val \
  --scorer-python "${FACE_QUALITY_SCORER_PYTHON}" --device cuda --batch-size 8 \
  --write --upload-per-image-asset --nonfatal
