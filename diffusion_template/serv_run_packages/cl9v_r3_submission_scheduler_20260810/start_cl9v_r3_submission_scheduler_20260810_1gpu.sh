#!/usr/bin/env bash
set -euo pipefail

# 10 Aug 2026 - User-authorized recurring submission scheduler. It never
# resubmits an accepted YAML and stops after both jobs receive immutable IDs.
TASK_OWNER_ROOT="/mnt/virtual_ai0001053-01309_SR006-nfs1/nasilaev"
TASK_STATE="${TASK_OWNER_ROOT}/analysis_sidecars/cl9v_r3_submission_scheduler_20260810"
TASK_MLS="${TASK_OWNER_ROOT}/conda_env/photomaker_NS/bin/mls"
TASK_PYTHON="${TASK_OWNER_ROOT}/conda_env/photomaker_NS/bin/python"
TASK_SMALLFACE_YAML="${TASK_OWNER_ROOT}/rsrch_test/diffusion_template/serv_run_packages/cl9v_smallface_roi_20260810_r3/run_cl9v_smallface_roi_20260810_r3_1gpu.yaml"
TASK_MARION_YAML="${TASK_OWNER_ROOT}/rsrch_test/diffusion_template/serv_run_packages/cl9v_marion_occlusion_20260810_r3/run_cl9v_marion_occlusion_20260810_r3_1gpu.yaml"
TASK_SMALLFACE_JOB_FILE="${TASK_STATE}/smallface.accepted_job_id"
TASK_SMALLFACE_TERMINAL_FILE="${TASK_STATE}/smallface.terminal_status"
TASK_GPU_CEILING=10
TASK_INTERVAL_SECONDS=1800
TASK_TARGET_LOCAL="2026-08-10T23:00:00"

mkdir -p "${TASK_STATE}/attempts"
test -x "${TASK_MLS}"
test -x "${TASK_PYTHON}"
test -s "${TASK_SMALLFACE_YAML}"
test -s "${TASK_MARION_YAML}"

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" \
    | tee -a "${TASK_STATE}/scheduler.log"
}

extract_job_id() {
  "${TASK_PYTHON}" -c '
import re, sys
text = sys.stdin.read()
matches = re.findall(r"[\"\x27]job_name[\"\x27]\s*:\s*[\"\x27]([^\"\x27]+)", text)
if matches:
    print(matches[-1])
'
}

project_gpu_count() {
  local running pending
  running="$("${TASK_MLS}" job list -s Running -l 100 -O json)"
  pending="$("${TASK_MLS}" job list -s Pending -l 100 -O json)"
  printf '%s\n%s' "${running}" "${pending}" | "${TASK_PYTHON}" -c '
import json, sys
text = sys.stdin.read()
decoder = json.JSONDecoder()
position = 0
total = 0
while position < len(text):
    while position < len(text) and text[position].isspace():
        position += 1
    if position >= len(text):
        break
    value, position = decoder.raw_decode(text, position)
    total += sum(
        int(job.get("gpu_count", 0))
        for job in value.get("jobs", [])
        if "#nasilaev" in str(job.get("job_desc", ""))
    )
print(total)
'
}

smallface_terminal_status() {
  local job_id output rc
  [[ -s "${TASK_SMALLFACE_JOB_FILE}" ]] || return 1
  job_id="$(<"${TASK_SMALLFACE_JOB_FILE}")"
  set +e
  output="$("${TASK_MLS}" job status "${job_id}" -O json 2>/dev/null)"
  rc=$?
  set -e
  [[ "${rc}" -eq 0 ]] || return 1
  printf '%s\n' "${output}" | "${TASK_PYTHON}" -c '
import json, sys
payload = json.load(sys.stdin)
status = str(payload.get("status", "")).strip().lower()
if status in {"completed", "failed", "killed", "cancelled", "canceled"}:
    print(status)
else:
    raise SystemExit(1)
'
}

try_submit() {
  local label="$1"
  local yaml="$2"
  local accepted_file="${TASK_STATE}/${label}.accepted_job_id"
  if [[ -s "${accepted_file}" ]]; then
    return 0
  fi

  local attempt_time attempt_file output rc job_id
  attempt_time="$(date -u +%Y%m%dT%H%M%SZ)"
  attempt_file="${TASK_STATE}/attempts/${attempt_time}_${label}.log"
  set +e
  output="$("${TASK_MLS}" job submit --config "${yaml}" -O json 2>&1)"
  rc=$?
  set -e
  printf '%s\n' "${output}" > "${attempt_file}"
  job_id="$(printf '%s\n' "${output}" | extract_job_id)"
  if [[ -n "${job_id}" ]]; then
    printf '%s\n' "${job_id}" > "${accepted_file}"
    log "ACCEPTED label=${label} job=${job_id} rc=${rc}"
    return 0
  fi
  log "REJECTED label=${label} rc=${rc} detail=$(printf '%s' "${output}" | tail -1)"
  return 1
}

target_epoch="$("${TASK_PYTHON}" -c '
from datetime import datetime
from zoneinfo import ZoneInfo
value = datetime.fromisoformat("'"${TASK_TARGET_LOCAL}"'").replace(tzinfo=ZoneInfo("Europe/London"))
print(int(value.timestamp()))
')"
printf '%s\n' "${target_epoch}" > "${TASK_STATE}/target_epoch.txt"
printf '%s\n' "${TASK_INTERVAL_SECONDS}" > "${TASK_STATE}/interval_seconds.txt"
log "START target_local=${TASK_TARGET_LOCAL} timezone=Europe/London interval_seconds=${TASK_INTERVAL_SECONDS} ceiling=${TASK_GPU_CEILING}"

next_attempt_epoch="${target_epoch}"
while true; do
  if [[ -e "${TASK_STATE}/STOP" ]]; then
    log "STOP_REQUESTED"
    exit 0
  fi
  if [[ -s "${TASK_STATE}/smallface.accepted_job_id" && -s "${TASK_STATE}/marion.accepted_job_id" ]]; then
    log "COMPLETE both_jobs_accepted"
    date -u +%Y-%m-%dT%H:%M:%SZ > "${TASK_STATE}/complete_at.txt"
    exit 0
  fi

  now_epoch="$(date +%s)"
  # AICODE-NOTE: The GPU worker cannot be mutated after MLS starts it. Polling
  # its terminal state lets Marion submit on allocation release without an
  # unsafe restart or waiting for the next fixed half-hour boundary.
  if [[ ! -s "${TASK_SMALLFACE_TERMINAL_FILE}" ]]; then
    dependency_status="$(smallface_terminal_status || true)"
    if [[ -n "${dependency_status}" ]]; then
      printf '%s\n' "${dependency_status}" > "${TASK_SMALLFACE_TERMINAL_FILE}"
      log "DEPENDENCY_TERMINAL label=smallface status=${dependency_status}; triggering immediate Marion attempt"
      next_attempt_epoch="${now_epoch}"
    fi
  fi
  if (( now_epoch >= next_attempt_epoch )); then
    current_gpus="$(project_gpu_count)"
    remaining=0
    [[ -s "${TASK_STATE}/smallface.accepted_job_id" ]] || remaining=$((remaining + 1))
    [[ -s "${TASK_STATE}/marion.accepted_job_id" ]] || remaining=$((remaining + 1))
    log "ATTEMPT project_gpus=${current_gpus} remaining_requests=${remaining} ceiling=${TASK_GPU_CEILING}"
    if (( current_gpus + remaining <= TASK_GPU_CEILING )); then
      try_submit smallface "${TASK_SMALLFACE_YAML}" || true
      try_submit marion "${TASK_MARION_YAML}" || true
    else
      log "SKIPPED_INTERNAL_CEILING project_gpus=${current_gpus} remaining_requests=${remaining}"
    fi
    now_epoch="$(date +%s)"
    if (( now_epoch < target_epoch )); then
      next_attempt_epoch="${target_epoch}"
    else
      next_attempt_epoch=$((target_epoch + ((now_epoch - target_epoch) / TASK_INTERVAL_SECONDS + 1) * TASK_INTERVAL_SECONDS))
    fi
    printf '%s\n' "${next_attempt_epoch}" > "${TASK_STATE}/next_attempt_epoch.txt"
  fi
  sleep 30
done
