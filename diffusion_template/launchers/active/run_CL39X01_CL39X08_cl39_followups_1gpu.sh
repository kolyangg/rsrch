#!/usr/bin/env bash
# CL39-X01..X08: isolated CL39 successors on fixed manual_val96.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/launchers/lib/prepare_comet_record.sh"

: "${RUN_NAME:?Set the unique CL39-X run name}"
: "${CONFIG_NAME:?Set the matching config name}"
: "${EXPERIMENT_SPEC_PATH:?Set the experiment JSON path}"
: "${COSMIC_LARGE_MANIFEST:?Set the Cosmic manifest}"
: "${COSMIC_LARGE_ROOT:?Set the Cosmic image root}"
: "${COMET_API_KEY:?Load COMET_API_KEY from .env}"
: "${FACE_QUALITY_SCORER_PYTHON:?Set PyIQA scorer Python}"
: "${SUBJECT_V2_ID_EMBEDS:?Set sealed subject-v2 embeddings}"
if [[ "$#" -ne 0 ]]; then echo "CL39-X launchers reject ad-hoc Hydra overrides." >&2; exit 2; fi
case "${CONFIG_NAME}" in
  CL39X01_cosmic_valid_key_attention_24k|CL39X02_cosmic_cycle_confidence_24k|\
  CL39X03_cosmic_stage_split_ot_transport_24k|CL39X04_cosmic_small_face_roi_route_24k|\
  CL39X05_cosmic_automask_os_24k|CL39X06_cosmic_counterfactual_reference_24k|\
  CL39X07_cosmic_intrinsic_id_sidecar_24k|CL39X08_cosmic_global_local_balance_24k) ;;
  *) echo "Unapproved CL39-X config: ${CONFIG_NAME}" >&2; exit 2 ;;
esac
test -s "${COSMIC_LARGE_MANIFEST}" && test -d "${COSMIC_LARGE_ROOT}"
if [[ "${CONFIG_NAME}" == CL39X05_* ]]; then
  : "${AUTOMASK_OS_CACHE_ROOT:?Set complete X05 training cache}"
  : "${AUTOMASK_OS_VAL_CACHE_ROOT:?Set complete X05 validation-reference cache}"
  test -s "${AUTOMASK_OS_CACHE_ROOT}/manifest.json"
  test -s "${AUTOMASK_OS_VAL_CACHE_ROOT}/manifest.json"
  python - "${AUTOMASK_OS_CACHE_ROOT}/manifest.json" "${AUTOMASK_OS_VAL_CACHE_ROOT}/manifest.json" <<'PY'
import json, sys
for path in sys.argv[1:]:
    value = json.load(open(path, encoding="utf-8"))
    if value.get("policy_version") != "automask_os_v1" or not value.get("complete"):
        raise SystemExit(f"Incomplete or incompatible AutoMask-OS manifest: {path}")
    if value.get("failures"):
        raise SystemExit(f"AutoMask-OS manifest contains failures: {path}")
validation = json.load(open(sys.argv[2], encoding="utf-8"))
training = json.load(open(sys.argv[1], encoding="utf-8"))
if training.get("recipe_kind") != "cl39x05_cosmic_raw_sources_v1":
    raise SystemExit("AutoMask-OS training recipe mismatch")
if validation.get("recipe_kind") != "cl39x05_manual_val_references_v1":
    raise SystemExit("AutoMask-OS validation-reference recipe mismatch")
PY
fi
python tools/validate_CL39X01_CL39X08_config.py \
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
if [[ "${CL39X_ONEBATCH_SMOKE:-0}" == 1 ]]; then
  # Operational gate only: one unchanged 12-item step-zero validation batch,
  # followed by two real optimizer steps and no periodic validation.
  OVERRIDES+=(
    datasets.val.manual_val.limit=12
    trainer.face_quality.expected_images=12
    trainer.epoch_len=2 trainer.n_epochs=1
    trainer.validation_interval_steps=0 trainer.save_period=999
    weights_only_save_period=0
  )
fi
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
if [[ "${CL39X_ONEBATCH_SMOKE:-0}" == 1 ]]; then
  EXPECTED_STEPS=0
  EXPECTED_IMAGES=12
else
  EXPECTED_STEPS="0,2000,4000,6000,8000,10000,12000,14000,16000,18000,20000,22000,24000"
  EXPECTED_IMAGES=96
fi
"${FACE_QUALITY_SCORER_PYTHON}" tools/comet/finalize_deferred_face_quality.py \
  --run-dir "${ROOT_DIR}/saved/${RUN_NAME}" --expected-project aug-large-ds \
  --expected-steps "${EXPECTED_STEPS}" --images-per-step "${EXPECTED_IMAGES}" \
  --partition manual_val --scorer-python "${FACE_QUALITY_SCORER_PYTHON}" \
  --device cuda --batch-size 8 --write --upload-per-image-asset --nonfatal
