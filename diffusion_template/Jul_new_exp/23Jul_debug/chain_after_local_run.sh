#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WATCH_PID="${1:?usage: chain_after_local_run.sh LOCAL_LAUNCHER_PID ARCHITECTURE}"
ARCHITECTURE="${2:?usage: chain_after_local_run.sh LOCAL_LAUNCHER_PID ARCHITECTURE}"

if [[ ! -r "/proc/${WATCH_PID}/cmdline" ]]; then
    echo "Watched PID ${WATCH_PID} is not running" >&2
    exit 2
fi
WATCH_COMMAND="$(tr '\0' ' ' <"/proc/${WATCH_PID}/cmdline")"
case "${WATCH_COMMAND}" in
    *"${HERE}/launch_training.py"*) ;;
    *"schedule_after_production_validation.sh"*) ;;
    *"chain_after_local_run.sh"*) ;;
    *)
        echo "Refusing to watch non-local experiment PID ${WATCH_PID}: ${WATCH_COMMAND}" >&2
        exit 2
        ;;
esac

echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') waiting for local PID ${WATCH_PID}"
while kill -0 "${WATCH_PID}" 2>/dev/null; do
    sleep 10
done

PRODUCTION_PID="$(
    pgrep -f '/home/niko/miniconda3/envs/photomaker_NS/bin/python -u train.py.*writer.run_name=ba_N3a_new1_1gpu' \
        | head -n 1 || true
)"
if [[ -z "${PRODUCTION_PID}" ]]; then
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') production absent; launching ${ARCHITECTURE}"
    exec "${HERE}/run_architecture.sh" "${ARCHITECTURE}"
fi

PRODUCTION_MEMORY="$(
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits \
        | awk -F',' -v wanted="${PRODUCTION_PID}" '
            {gsub(/ /, "", $1); gsub(/ /, "", $2)}
            $1 == wanted {print $2; found=1}
            END {if (!found) print 0}
        '
)"
if (( PRODUCTION_MEMORY > 0 && PRODUCTION_MEMORY <= 38000 )); then
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') production training at ${PRODUCTION_MEMORY} MiB; launching ${ARCHITECTURE}"
    exec "${HERE}/run_architecture.sh" "${ARCHITECTURE}"
fi

echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') production memory ${PRODUCTION_MEMORY} MiB; deferring ${ARCHITECTURE}"
exec "${HERE}/schedule_after_production_validation.sh" "${ARCHITECTURE}"
