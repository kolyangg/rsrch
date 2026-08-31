#!/usr/bin/env bash
# Four independent post-CL39 architecture leaves on one Serv A100 each.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/launchers/lib/prepare_comet_record.sh"

: "${RUN_NAME:?Set the unique run name}"
: "${CONFIG_NAME:?Set the matching config name}"
: "${EXPERIMENT_SPEC_PATH:?Set the immutable experiment record}"
: "${COSMIC_LARGE_MANIFEST:?Set the Cosmic manifest}"
: "${COSMIC_LARGE_ROOT:?Set the Cosmic image root}"
: "${COMET_API_KEY:?Load COMET_API_KEY from .env}"
: "${FACE_QUALITY_SCORER_PYTHON:?Set PyIQA scorer Python}"
: "${SUBJECT_V2_ID_EMBEDS:?Set sealed subject-v2 embeddings}"
if [[ "$#" -ne 0 ]]; then echo "CL39N launchers reject Hydra overrides" >&2; exit 2; fi
case "${CONFIG_NAME}" in
  CL39N6R_cosmic_up1_low_pruned_24k|CL39N7_cosmic_posterior_null_router_24k|\
  CL39N8_cosmic_native_orthogonal_highband_24k|CL39N9_cosmic_intrinsic_id_sidecar_24k) ;;
  *) echo "Unapproved CL39N config: ${CONFIG_NAME}" >&2; exit 2 ;;
esac
test -s "${COSMIC_LARGE_MANIFEST}" && test -d "${COSMIC_LARGE_ROOT}"
if [[ "${CONFIG_NAME}" == CL39N6R_* ]]; then
  : "${CL39N6R_CONFIRMATION_ARTIFACT:?N6R requires the independent seed-1 gate}"
  python - "${CL39N6R_CONFIRMATION_ARTIFACT}" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
expected = "858c4663083ccffbd461e94215d4e9951f2765b59b4f49ce454de92c5910904f"
if value.get("status") != "pass" or value.get("map_sha256") != expected:
    raise SystemExit("CL39N6R seed-1 confirmation did not pass for the sealed map")
PY
fi

python tools/validate_CL39N6R_CL39N9_config.py \
  --config-name "${CONFIG_NAME}" --run-name "${RUN_NAME}" \
  --experiment-spec "${EXPERIMENT_SPEC_PATH}"
mkdir -p "${ROOT_DIR}/logs/preflight"
python tools/datasets/preflight_cosmic_cl.py --config-name "${CONFIG_NAME}" \
  --sample-count "${COSMIC_PREFLIGHT_SAMPLES:-64}" \
  --output "${ROOT_DIR}/logs/preflight/${RUN_NAME}.json"
prepare_comet_record "${ROOT_DIR}" "${RUN_NAME}" "${EXPERIMENT_SPEC_PATH}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HYDRA_FULL_ERROR=1 ACCELERATE_LOG_LEVEL=error TRANSFORMERS_VERBOSITY=error
export DIFFUSERS_VERBOSITY=error COMET_DISABLE_AUTO_LOGGING=1 COMET_LOGGING_CONSOLE=ERROR
export ACCELERATE_NUM_PROCESSES=1
OVERRIDES=("metrics.id_sim_subject_v2.id_embeds_pth=${SUBJECT_V2_ID_EMBEDS}")
if [[ -n "${PM_PATH:-}" ]]; then OVERRIDES+=("model.photomaker_path=${PM_PATH}"); fi
accelerate launch --config_file=src/configs/ddp/accelerate.yaml --num_processes=1 \
  train.py "--config-name=${CONFIG_NAME}" writer=cometml \
  "writer.run_name=${RUN_NAME}" writer.project_name=aug-large-ds "${OVERRIDES[@]}" &
TRAIN_PID=$!
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
  kill "${TRAIN_PID}" 2>/dev/null || true; wait "${TRAIN_PID}" || true; exit 78
fi
wait "${TRAIN_PID}"
"${FACE_QUALITY_SCORER_PYTHON}" tools/comet/finalize_deferred_face_quality.py \
  --run-dir "${ROOT_DIR}/saved/${RUN_NAME}" --expected-project aug-large-ds \
  --expected-steps 0,2000,4000,6000,8000,10000,12000,14000,16000,18000,20000,22000,24000 \
  --images-per-step 96 --partition manual_val \
  --scorer-python "${FACE_QUALITY_SCORER_PYTHON}" --device cuda --batch-size 8 \
  --write --upload-per-image-asset --nonfatal
