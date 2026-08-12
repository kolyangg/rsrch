#!/usr/bin/env bash
set -euo pipefail

# 11 Aug 2026 - User-authorized 30-minute capacity monitor for one chained
# one-A100 validation job. It exits permanently after the first accepted MLS
# response and never submits while this project's normal six-GPU ceiling is full.
TASK_PROJECT_ROOT="/home/kolyangg/rsrch_apr_test/diffusion_template"
TASK_STATE="${TASK_PROJECT_ROOT}/analysis_sidecars/cl9v_validation_chain_submission_scheduler_20260811"
TASK_SERV_HELPER="${TASK_PROJECT_ROOT}/../local_scripts/serv_job.py"
TASK_REMOTE_YAML="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test/diffusion_template/serv_run_packages/cl9v_validation_chain_20260811_r4/run_cl9v_validation_chain_20260811_r4_1gpu.yaml"
TASK_INTERVAL_SECONDS=1800
TASK_GPU_CEILING=6

mkdir -p "${TASK_STATE}/attempts"
test -s "${TASK_SERV_HELPER}"

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" \
    | tee -a "${TASK_STATE}/scheduler.log"
}

extract_job_id() {
  python3 -c '
import re, sys
text = sys.stdin.read()
matches = re.findall(r"lm-mpi-job-[A-Za-z0-9-]+", text)
if matches:
    print(matches[-1])
'
}

project_gpu_count() {
  ssh -T -o BatchMode=yes -o ConnectTimeout=20 serv 'bash -ic '\''
conda activate /mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/conda_env/photomaker_NS/ &&
cd /mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev/rsrch_test &&
python - <<"PY"
import json
import subprocess
total = 0
for status in ("Running", "Pending"):
    output = subprocess.check_output(
        ["mls", "job", "list", "-s", status, "-l", "100", "-O", "json"],
        text=True,
    )
    payload = json.loads(output)
    total += sum(
        int(job.get("gpu_count", 0))
        for job in payload.get("jobs", [])
        if "#nasilaev" in str(job.get("job_desc", ""))
    )
print(total)
PY
'\'''
}

attempt_once() {
  local attempt_time attempt_file output rc job_id
  attempt_time="$(date -u +%Y%m%dT%H%M%SZ)"
  attempt_file="${TASK_STATE}/attempts/${attempt_time}.log"
  set +e
  output="$(
    cd "${TASK_PROJECT_ROOT}" &&
    python3 "${TASK_SERV_HELPER}" submit "${TASK_REMOTE_YAML}" \
      --comment "CL9 24k chained ROI multiseed, precise occluder, and Marion roll validation"
  2>&1)"
  rc=$?
  set -e
  printf '%s\n' "${output}" > "${attempt_file}"
  job_id="$(printf '%s\n' "${output}" | extract_job_id)"
  if [[ -n "${job_id}" ]]; then
    printf '%s\n' "${job_id}" > "${TASK_STATE}/accepted_job_id"
    date -u +%Y-%m-%dT%H:%M:%SZ > "${TASK_STATE}/accepted_at.txt"
    log "ACCEPTED job=${job_id} rc=${rc}"
    return 0
  fi
  log "REJECTED rc=${rc} attempt_file=${attempt_file}"
  return 1
}

printf '%s\n' "${TASK_INTERVAL_SECONDS}" > "${TASK_STATE}/interval_seconds.txt"
printf '%s\n' "${TASK_GPU_CEILING}" > "${TASK_STATE}/gpu_ceiling.txt"
log "START interval_seconds=${TASK_INTERVAL_SECONDS} ceiling=${TASK_GPU_CEILING}"

next_check_epoch="$(date +%s)"
while true; do
  if [[ -e "${TASK_STATE}/STOP" ]]; then
    log "STOP_REQUESTED"
    exit 0
  fi
  if [[ -s "${TASK_STATE}/accepted_job_id" ]]; then
    log "COMPLETE already_accepted job=$(<"${TASK_STATE}/accepted_job_id")"
    exit 0
  fi
  now_epoch="$(date +%s)"
  if (( now_epoch >= next_check_epoch )); then
    current_gpus="$(project_gpu_count)"
    checked_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '%s %s\n' "${checked_at}" "${current_gpus}" \
      >> "${TASK_STATE}/capacity_checks.tsv"
    log "CHECK project_gpus=${current_gpus} ceiling=${TASK_GPU_CEILING}"
    if (( current_gpus + 1 <= TASK_GPU_CEILING )); then
      if attempt_once; then
        exit 0
      fi
    else
      log "SKIPPED_INTERNAL_CEILING project_gpus=${current_gpus} request=1"
    fi
    next_check_epoch=$((now_epoch + TASK_INTERVAL_SECONDS))
    printf '%s\n' "${next_check_epoch}" > "${TASK_STATE}/next_check_epoch.txt"
  fi
  sleep 30
done
