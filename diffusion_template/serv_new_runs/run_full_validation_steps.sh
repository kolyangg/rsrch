#!/usr/bin/env bash
# Full-val inference for selected intermediate checkpoints of one saved run.
#
# Usage:
#   bash serv_new_runs/run_full_validation_steps.sh <run_name> <step> [<step> ...]
#
# Example:
#   BATCH_SIZE=4 bash serv_new_runs/run_full_validation_steps.sh ba_longrun_N17 8000 10000 12000 14000 16000
#
# For saved/<run_name>/config.yaml with trainer.epoch_len=2000, step 16000 maps to
# saved/<run_name>/weights-epoch8.pth. Images are saved to:
#   full_validation_results/<run_name>_step<step>/
# Metrics are appended to:
#   full_validation_results/metrics_<run_name>_steps.json

set -uo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash $0 <run_name> <step> [<step> ...]" >&2
  exit 2
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
cd "${REPO}"

RUN="$1"
shift
STEPS=("$@")

RESULTS_DIR="${RESULTS_DIR:-${REPO}/full_validation_results}"
RUN_DIR="${REPO}/saved/${RUN}"
METRICS_JSON="${METRICS_JSON:-${RESULTS_DIR}/metrics_${RUN}_steps.json}"
REFS_DIR="${REFS_DIR:-../dataset_full/val_dataset/references}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EXPECTED_IMAGES="${EXPECTED_IMAGES:-96}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

PM_PATH="${PM_PATH:-/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/checkpoints/PhotoMaker-V2/photomaker-v2.bin}"
export PM_PATH

mkdir -p "${RESULTS_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${RESULTS_DIR}/run_full_validation_steps_${RUN}_${TS}.log"

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${LOG}"; }

if [[ ! -d "${RUN_DIR}" ]]; then
  log "ERROR: saved run not found: saved/${RUN}"
  exit 1
fi
if [[ ! -f "${RUN_DIR}/config.yaml" ]]; then
  log "ERROR: missing ${RUN_DIR}/config.yaml"
  exit 1
fi

epoch_len="$(grep -E '^\s*epoch_len:' "${RUN_DIR}/config.yaml" | head -1 | grep -oE '[0-9]+' | head -1)"
epoch_len="${epoch_len:-1000}"

log "intermediate full-val start | run=${RUN} | epoch_len=${epoch_len} | steps=${STEPS[*]}"
log "results -> ${RESULTS_DIR}/<run>_step<step> | metrics -> ${METRICS_JSON} | batch_size=${BATCH_SIZE}"

overall=0
for step in "${STEPS[@]}"; do
  if ! [[ "${step}" =~ ^[0-9]+$ ]]; then
    log "SKIP step=${step}: not an integer"
    overall=1
    continue
  fi
  if (( step % epoch_len != 0 )); then
    log "SKIP step=${step}: not divisible by epoch_len=${epoch_len}"
    overall=1
    continue
  fi

  epoch=$((step / epoch_len))
  ckpt="${RUN_DIR}/weights-epoch${epoch}.pth"
  if [[ ! -f "${ckpt}" ]]; then
    ckpt="${RUN_DIR}/checkpoint-epoch${epoch}.pth"
  fi
  if [[ ! -f "${ckpt}" ]]; then
    log "SKIP step=${step}: no weights/checkpoint for epoch ${epoch} in saved/${RUN}"
    overall=1
    continue
  fi

  run_key="${RUN}_step${step}"
  out_dir="${RESULTS_DIR}/${run_key}"
  have_imgs="$(ls -1 "${out_dir}"/*.png 2>/dev/null | grep -vc '^.*/_' || true)"
  if [[ "${have_imgs}" -ge "${EXPECTED_IMAGES}" ]]; then
    log "SKIP ${run_key}: ${have_imgs}/${EXPECTED_IMAGES} images already present"
  else
    log "START ${run_key} | epoch=${epoch} | ckpt=$(basename "${ckpt}")"
    ACCELERATE_LOG_LEVEL=error TRANSFORMERS_VERBOSITY=error DIFFUSERS_VERBOSITY=error \
    PYTHONWARNINGS="ignore::FutureWarning" HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    "${PYTHON_BIN}" infer.py --config-name inference/full_val \
        saved_checkpoint="saved/${RUN}/$(basename "${ckpt}")" \
        output_dir="${out_dir}" \
        batch_size="${BATCH_SIZE}" >>"${LOG}" 2>&1
    rc=$?
    if [[ ${rc} -ne 0 ]]; then
      log "FAIL ${run_key}: infer.py rc=${rc} (see ${LOG})"
      overall=1
      continue
    fi
  fi

  "${PYTHON_BIN}" scripts/full_val_metrics.py \
      --out-dir "${out_dir}" \
      --refs-dir "${REFS_DIR}" \
      --run "${run_key}" \
      --epoch "${epoch}" \
      --step "${step}" \
      --json "${METRICS_JSON}" \
      --checkpoint "saved/${RUN}/$(basename "${ckpt}")" >>"${LOG}" 2>&1 || {
        log "WARN ${run_key}: metric computation failed (images may be present)"
        overall=1
        continue
      }

  mean="$("${PYTHON_BIN}" -c "import json; d=json.load(open('${METRICS_JSON}')); print(d.get('${run_key}',{}).get('mean_id_sim'))" 2>/dev/null || echo '?')"
  log "DONE ${run_key} | mean_id_sim=${mean}"
done

log "intermediate full-val finished | overall_status=${overall}"
log "log: ${LOG}"
log "metrics JSON: ${METRICS_JSON}"
exit "${overall}"
