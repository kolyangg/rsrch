#!/usr/bin/env bash
# CL38-CL45: independent CL27 architecture/optimization arms on manual_val96.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/launchers/lib/prepare_comet_record.sh"

: "${RUN_NAME:?Set the unique CL38-CL45 run name}"
: "${CONFIG_NAME:?Set the matching config name}"
: "${EXPERIMENT_SPEC_PATH:?Set the immutable experiment JSON path}"
: "${COSMIC_LARGE_MANIFEST:?Set the filtered Cosmic manifest}"
: "${COSMIC_LARGE_ROOT:?Set the Cosmic image root}"
: "${COMET_API_KEY:?Load COMET_API_KEY from diffusion_template/.env}"
: "${FACE_QUALITY_SCORER_PYTHON:?Set the PyIQA scorer interpreter}"
: "${SUBJECT_V2_ID_EMBEDS:?Set sealed subject-v2 embeddings}"

if [[ "$#" -ne 0 ]]; then
  echo "CL38-CL45 launchers reject ad-hoc Hydra overrides." >&2
  exit 2
fi
case "${CONFIG_NAME}" in
  CL38_cosmic_visibility_ownership_v2_24k|\
  CL39_cosmic_null_key_confidence_router_24k|\
  CL40_cosmic_identity_motion_projector_24k|\
  CL41_cosmic_landmark_canonical_kv_24k|\
  CL42_cosmic_component_token_memory_24k|\
  CL43_cosmic_id_adaptive_modulation_24k|\
  CL44_cosmic_semantic_window_gate_24k|\
  CL45_cosmic_ba_pcgrad_24k) ;;
  *) echo "Unapproved CL38-CL45 config: ${CONFIG_NAME}" >&2; exit 2 ;;
esac

test -s "${COSMIC_LARGE_MANIFEST}"
test -d "${COSMIC_LARGE_ROOT}"
python tools/validate_CL38_CL45_config.py \
  --config-name "${CONFIG_NAME}" --run-name "${RUN_NAME}" \
  --experiment-spec "${EXPERIMENT_SPEC_PATH}"
mkdir -p "${ROOT_DIR}/logs/preflight"
python tools/datasets/preflight_cosmic_cl.py \
  --config-name "${CONFIG_NAME}" --sample-count "${COSMIC_PREFLIGHT_SAMPLES:-64}" \
  --output "${ROOT_DIR}/logs/preflight/${RUN_NAME}.json"
prepare_comet_record "${ROOT_DIR}" "${RUN_NAME}" "${EXPERIMENT_SPEC_PATH}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HYDRA_FULL_ERROR=1 ACCELERATE_LOG_LEVEL=error TRANSFORMERS_VERBOSITY=error
export DIFFUSERS_VERBOSITY=error COMET_DISABLE_AUTO_LOGGING=1 COMET_LOGGING_CONSOLE=ERROR
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
for _ in $(seq 1 300); do
  if [[ -s "${COMET_RECORD}" ]] && python - "${COMET_RECORD}" <<'PY'
import json, sys
key = (json.load(open(sys.argv[1], encoding="utf-8")).get("comet") or {}).get("experiment_key")
raise SystemExit(0 if isinstance(key, str) and len(key) == 32 else 1)
PY
  then echo "COMET_STARTUP_VERIFIED ${COMET_RECORD}"; COMET_READY=1; break; fi
  if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then wait "${TRAIN_PID}"; exit $?; fi
  sleep 2
done
if [[ "${COMET_READY:-0}" -ne 1 ]]; then
  echo "Comet immutable key was not registered within 10 minutes." >&2
  kill "${TRAIN_PID}" 2>/dev/null || true; wait "${TRAIN_PID}" || true; exit 78
fi
set +e; wait "${TRAIN_PID}"; TRAIN_STATUS=$?; set -e
if [[ "${TRAIN_STATUS}" -ne 0 ]]; then exit "${TRAIN_STATUS}"; fi

"${FACE_QUALITY_SCORER_PYTHON}" tools/comet/finalize_deferred_face_quality.py \
  --run-dir "${ROOT_DIR}/saved/${RUN_NAME}" --expected-project aug-large-ds \
  --expected-steps "0,2000,4000,6000,8000,10000,12000,14000,16000,18000,20000,22000,24000" \
  --images-per-step 96 --partition manual_val \
  --scorer-python "${FACE_QUALITY_SCORER_PYTHON}" --device cuda --batch-size 8 \
  --write --upload-per-image-asset --nonfatal
