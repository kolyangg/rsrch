#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${1:?usage: schedule_validation_after_production.sh RUN_DIR [STEPS] [MODES]}"
STEPS="${2:-0,200,400,600}"
MODES="${3:-canonical50,pmControl50}"
STATUS_DIR="${HERE}/scheduler"
mkdir -p "${STATUS_DIR}"

RUN_TAG="$(basename -- "${RUN_DIR}")"
LOCK_DIR="${STATUS_DIR}/validation_${RUN_TAG}.lock"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
    echo "A validation scheduler is already active: ${LOCK_DIR}" >&2
    exit 2
fi
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

timestamp() {
    date -u '+%Y-%m-%dT%H:%M:%SZ'
}

main_pid() {
    pgrep -f '/home/niko/miniconda3/envs/photomaker_NS/bin/python -u train.py.*writer.run_name=ba_N3a_new1_1gpu' \
        | head -n 1 || true
}

pid_memory_mib() {
    local pid="$1"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits \
        | awk -F',' -v wanted="${pid}" '
            {gsub(/ /, "", $1); gsub(/ /, "", $2)}
            $1 == wanted {print $2; found=1}
            END {if (!found) print 0}
        '
}

PID="$(main_pid)"
if [[ -n "${PID}" ]]; then
    echo "$(timestamp) watching production PID ${PID}; waiting for validation to finish"
    SAW_VALIDATION=0
    LOW_COUNT=0
    while kill -0 "${PID}" 2>/dev/null; do
        MEMORY="$(pid_memory_mib "${PID}")"
        if (( MEMORY >= 42000 )); then
            SAW_VALIDATION=1
            LOW_COUNT=0
        elif (( SAW_VALIDATION == 1 && MEMORY > 0 && MEMORY <= 38000 )); then
            LOW_COUNT=$((LOW_COUNT + 1))
            if (( LOW_COUNT >= 3 )); then
                echo "$(timestamp) production training window restored at ${MEMORY} MiB"
                break
            fi
        else
            LOW_COUNT=0
        fi
        sleep 10
    done
fi

echo "$(timestamp) validating ${RUN_TAG}; steps=${STEPS}; modes=${MODES}"
exec "${HERE}/run_validation_suite.sh" \
    "${RUN_DIR}" \
    --steps "${STEPS}" \
    --modes "${MODES}"
