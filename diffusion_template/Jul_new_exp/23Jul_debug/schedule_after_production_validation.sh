#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ARCHITECTURE="${1:-E00_control}"
STATUS_DIR="${HERE}/scheduler"
mkdir -p "${STATUS_DIR}"

LOCK_DIR="${STATUS_DIR}/${ARCHITECTURE}.lock"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
    echo "A scheduler for ${ARCHITECTURE} is already active: ${LOCK_DIR}" >&2
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
if [[ -z "${PID}" ]]; then
    echo "$(timestamp) production run not found; launching ${ARCHITECTURE}"
    exec "${HERE}/run_architecture.sh" "${ARCHITECTURE}"
fi

echo "$(timestamp) watching production PID ${PID}; waiting for its next validation"
SAW_VALIDATION=0
LOW_COUNT=0
while true; do
    if ! kill -0 "${PID}" 2>/dev/null; then
        echo "$(timestamp) production PID exited; launching ${ARCHITECTURE}"
        break
    fi

    MEMORY="$(pid_memory_mib "${PID}")"
    # The resumed production job currently uses ~34.7 GiB while training and
    # ~45.7 GiB in its first validation batch (later batches can peak higher).
    if (( MEMORY >= 42000 )); then
        if (( SAW_VALIDATION == 0 )); then
            echo "$(timestamp) production validation detected at ${MEMORY} MiB"
        fi
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

echo "$(timestamp) launching ${ARCHITECTURE}"
exec "${HERE}/run_architecture.sh" "${ARCHITECTURE}"
