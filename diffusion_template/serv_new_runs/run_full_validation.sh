#!/usr/bin/env bash
# Full-validation-set inference for every saved run.
#
# For each run in saved/<run>/ this:
#   1. picks the LATEST weights-epochN.pth  (epoch = N; step = N * epoch_len from the run's config),
#   2. runs infer.py on the FULL 8-identity validation set (references/, 96 images) via the
#      inference/full_val.yaml base config,
#   3. saves the 96 images to full_validation_results/<run>/,
#   4. computes id-sim metrics (InsightFace, same method as all project analysis) and appends them
#      to full_validation_results/metrics.json (with epoch + step per run).
#
# Progress, per-run timing and an ETA are printed to the terminal AND logged to
# full_validation_results/run_full_validation_<timestamp>.log.
#
# Resumable: a run whose output dir already has >= EXPECTED_IMAGES images AND an entry in
# metrics.json is skipped, so re-running continues where it left off.
#
# USAGE (from repo root, photomaker_NS venv active):
#   cd /workspace/rsrch/diffusion_template
#   tmux new -s fullval           # long job; survives disconnects
#   bash serv_new_runs/run_full_validation.sh
#
# Speed: 96 images x 50 steps per run x ~11 runs. With BATCH_SIZE=4 and deterministic pm96 gen
# boxes (no per-image YOLO/PM preview) this is much faster than bs=1 auto-bbox. Raise/lower
# BATCH_SIZE via env (bs=4 ~= 35/46GB; try 6 if you want more utilization).

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
cd "${REPO}"

# ---- config ----
RESULTS_DIR="${REPO}/full_validation_results"
METRICS_JSON="${RESULTS_DIR}/metrics.json"
REFS_DIR="../dataset_full/val_dataset/references"       # ref images (dataset restricts to pm96 ids)
EXPECTED_IMAGES=96                                      # 8 pm96 ids x 12 prompts
# Inference has no gradients; bs=1 used only ~17/46GB. bs=4 ~= 35GB. Raise/lower via env.
BATCH_SIZE="${BATCH_SIZE:-4}"

# PhotoMaker weights (same default as the training scripts; override by exporting PM_PATH).
PM_PATH="${PM_PATH:-/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/checkpoints/PhotoMaker-V2/photomaker-v2.bin}"
export PM_PATH

# Runs to validate, in order. Edit to add/remove.
RUNS=(
  ba_nr_alt_N3a ba_nr_alt_N4 ba_nr_alt_N5 ba_nr_blend_N6
  ba_coadapt_N10 ba_saonly_N11 ba_idembeds_N12 ba_idloss_N13
  ba_combo_N14 ba_saonly6k_N15 ba_idloss6k_N16
)

mkdir -p "${RESULTS_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${RESULTS_DIR}/run_full_validation_${TS}.log"

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${LOG}"; }
hms() { local s=$1; printf '%dh%02dm' $((s/3600)) $(((s%3600)/60)); }

log "full validation start | repo=${REPO} | batch_size=${BATCH_SIZE} | runs=${#RUNS[@]}"
log "results -> ${RESULTS_DIR} | metrics -> ${METRICS_JSON} | gen bboxes = pm96 (deterministic, 96 imgs)"

total=${#RUNS[@]}
done_count=0
sum_secs=0
overall=0

for i in "${!RUNS[@]}"; do
  run="${RUNS[$i]}"
  idx=$((i + 1))
  run_dir="${REPO}/saved/${run}"
  out_dir="${RESULTS_DIR}/${run}"

  echo | tee -a "${LOG}"
  if [[ ! -d "${run_dir}" ]]; then
    log "[${idx}/${total}] SKIP ${run}: saved/${run} not found"
    overall=1
    continue
  fi

  # latest weights-epochN.pth (highest N)
  ckpt="$(ls -1 "${run_dir}"/weights-epoch*.pth 2>/dev/null \
          | sed -E 's/.*weights-epoch([0-9]+)\.pth/\1 &/' | sort -n | tail -1 | awk '{print $2}')"
  if [[ -z "${ckpt}" ]]; then
    log "[${idx}/${total}] SKIP ${run}: no weights-epoch*.pth in saved/${run}"
    overall=1
    continue
  fi
  epoch="$(echo "${ckpt}" | sed -E 's/.*weights-epoch([0-9]+)\.pth/\1/')"
  epoch_len="$(grep -E '^\s*epoch_len:' "${run_dir}/config.yaml" 2>/dev/null | head -1 | grep -oE '[0-9]+' | head -1)"
  epoch_len="${epoch_len:-1000}"
  step=$((epoch * epoch_len))

  # resume: skip if already complete
  have_imgs="$(ls -1 "${out_dir}"/*.png 2>/dev/null | grep -vc '^.*/_' || true)"
  if [[ "${have_imgs}" -ge "${EXPECTED_IMAGES}" ]] && grep -q "\"${run}\"" "${METRICS_JSON}" 2>/dev/null; then
    log "[${idx}/${total}] SKIP ${run}: already done (${have_imgs} imgs + metrics present)"
    done_count=$((done_count + 1))
    continue
  fi

  # per-run ETA from average of completed runs
  eta="n/a"
  if [[ ${done_count} -gt 0 ]]; then
    avg=$((sum_secs / done_count))
    eta="$(hms $((avg * (total - idx + 1))))"
  fi
  log "[${idx}/${total}] START ${run} | epoch=${epoch} step=${step} | ckpt=$(basename "${ckpt}") | ETA_remaining≈${eta}"

  # id_embeds run needs the id_embeds conditioning at inference too
  extra=()
  if [[ "${run}" == "ba_idembeds_N12" ]]; then
    extra=( pipeline.face_embed_strategy=id_embeds model.use_id_embeds=true
            validation_args.face_embed_strategy=id_embeds )
    log "        (${run}: using face_embed_strategy=id_embeds at inference)"
  fi

  start_s=$(date +%s)
  ACCELERATE_LOG_LEVEL=error TRANSFORMERS_VERBOSITY=error DIFFUSERS_VERBOSITY=error \
  PYTHONWARNINGS="ignore::FutureWarning" HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 \
  python infer.py --config-name inference/full_val \
      saved_checkpoint="saved/${run}/$(basename "${ckpt}")" \
      output_dir="${out_dir}" \
      batch_size="${BATCH_SIZE}" \
      "${extra[@]}" >>"${LOG}" 2>&1
  rc=$?

  if [[ ${rc} -ne 0 ]]; then
    log "[${idx}/${total}] FAIL ${run}: infer.py rc=${rc} (see ${LOG}) — continuing"
    overall=1
    continue
  fi

  # metrics -> JSON (with epoch + step)
  python scripts/full_val_metrics.py \
      --out-dir "${out_dir}" --refs-dir "${REFS_DIR}" --run "${run}" \
      --epoch "${epoch}" --step "${step}" --json "${METRICS_JSON}" \
      --checkpoint "saved/${run}/$(basename "${ckpt}")" >>"${LOG}" 2>&1 || \
      log "[${idx}/${total}] WARN ${run}: metric computation failed (images saved; see ${LOG})"

  dur=$(( $(date +%s) - start_s ))
  sum_secs=$((sum_secs + dur))
  done_count=$((done_count + 1))
  mean="$(python -c "import json;d=json.load(open('${METRICS_JSON}'));print(d.get('${run}',{}).get('mean_id_sim'))" 2>/dev/null || echo '?')"
  log "[${idx}/${total}] DONE ${run} in $(hms ${dur}) | mean_id_sim=${mean}"
done

echo | tee -a "${LOG}"
log "full validation finished | completed=${done_count}/${total} | overall_status=${overall}"
log "metrics JSON: ${METRICS_JSON}"
exit "${overall}"
