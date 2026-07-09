#!/usr/bin/env bash
# Full-validation baseline for base PhotoMaker V2: same full_val dataset/seeds,
# no saved BA checkpoint, and branched attention disabled.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
cd "${REPO}"

RESULTS_DIR="${RESULTS_DIR:-${REPO}/full_validation_results}"
OUT_DIR="${OUT_DIR:-${RESULTS_DIR}/photomaker_baseline}"
METRICS_JSON="${METRICS_JSON:-${RESULTS_DIR}/metrics_photomaker_baseline.json}"
REFS_DIR="${REFS_DIR:-../dataset_full/val_dataset/references}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EXPECTED_IMAGES="${EXPECTED_IMAGES:-96}"
FORCE="${FORCE:-0}"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

PM_PATH="${PM_PATH:-/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/checkpoints/PhotoMaker-V2/photomaker-v2.bin}"
export PM_PATH

mkdir -p "${RESULTS_DIR}" "${OUT_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${RESULTS_DIR}/photomaker_baseline_full_validation_${TS}.log"

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${LOG}"; }

log "PhotoMaker baseline full-val start | out=${OUT_DIR} | batch_size=${BATCH_SIZE}"
log "PM_PATH=${PM_PATH}"
log "metrics -> ${METRICS_JSON}"

have_imgs="$(ls -1 "${OUT_DIR}"/*.png 2>/dev/null | grep -vc '^.*/_' || true)"
if [[ "${have_imgs}" -ge "${EXPECTED_IMAGES}" && "${FORCE}" != "1" ]]; then
  log "SKIP inference: ${have_imgs}/${EXPECTED_IMAGES} images already present. Set FORCE=1 to regenerate."
else
  ACCELERATE_LOG_LEVEL=error TRANSFORMERS_VERBOSITY=error DIFFUSERS_VERBOSITY=error \
  PYTHONWARNINGS="ignore::FutureWarning" HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  "${PYTHON_BIN}" infer.py --config-name inference/full_val \
      saved_checkpoint=null \
      output_dir="${OUT_DIR}" \
      batch_size="${BATCH_SIZE}" \
      validation_args.use_branched_attention=false \
      validation_args.face_embed_strategy=id \
      pipeline.face_embed_strategy=id \
      pipeline.use_id_embeds=false \
      model.use_id_embeds=false \
      disable_branched_sa=true \
      disable_branched_ca=true >>"${LOG}" 2>&1
  rc=$?
  if [[ ${rc} -ne 0 ]]; then
    log "FAIL: infer.py rc=${rc} (see ${LOG})"
    exit "${rc}"
  fi
fi

"${PYTHON_BIN}" scripts/full_val_metrics.py \
    --out-dir "${OUT_DIR}" \
    --refs-dir "${REFS_DIR}" \
    --run photomaker_baseline \
    --epoch 0 \
    --step 0 \
    --json "${METRICS_JSON}" \
    --checkpoint "PhotoMaker-V2:${PM_PATH}" >>"${LOG}" 2>&1
rc=$?
if [[ ${rc} -ne 0 ]]; then
  log "WARN: metric computation failed rc=${rc} (images may be present; see ${LOG})"
  exit "${rc}"
fi

mean="$("${PYTHON_BIN}" -c "import json; d=json.load(open('${METRICS_JSON}')); print(d.get('photomaker_baseline',{}).get('mean_id_sim'))" 2>/dev/null || echo '?')"
log "DONE photomaker_baseline | mean_id_sim=${mean}"
log "images: ${OUT_DIR}"
log "log: ${LOG}"
