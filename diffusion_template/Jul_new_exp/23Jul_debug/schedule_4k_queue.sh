#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCHEDULER_DIR="${HERE}/scheduler_4k"
mkdir -p "${SCHEDULER_DIR}"
STATUS="${SCHEDULER_DIR}/QUEUE_STATUS.md"

PAIRS=(
    "L4_O1_oneid_projection_alt L4_C1_large_projection_alt"
    "L4_O2_oneid_projection_blend20 L4_C2_large_projection_blend20"
    "L4_O3_oneid_ref_value_blend20 L4_C3_large_ref_value_blend20"
)

{
    echo "# 4k queue status"
    echo
    echo "Started: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
} >"${STATUS}"

pair_index=0
for pair in "${PAIRS[@]}"; do
    pair_index=$((pair_index + 1))
    read -r left right <<<"${pair}"
    left_log="${SCHEDULER_DIR}/${left}.log"
    right_log="${SCHEDULER_DIR}/${right}.log"
    "${HERE}/run_4k_arm.sh" "${left}" >"${left_log}" 2>&1 &
    left_pid=$!
    "${HERE}/run_4k_arm.sh" "${right}" >"${right_log}" 2>&1 &
    right_pid=$!
    {
        echo
        echo "## Pair ${pair_index}"
        echo
        echo "- $(date -u '+%Y-%m-%dT%H:%M:%SZ'): started \`${left}\` PID ${left_pid}"
        echo "- $(date -u '+%Y-%m-%dT%H:%M:%SZ'): started \`${right}\` PID ${right_pid}"
    } >>"${STATUS}"
    set +e
    wait "${left_pid}"
    left_status=$?
    wait "${right_pid}"
    right_status=$?
    set -e
    {
        echo "- $(date -u '+%Y-%m-%dT%H:%M:%SZ'): pair finished; statuses ${left_status}/${right_status}"
    } >>"${STATUS}"
    if (( left_status != 0 || right_status != 0 )); then
        echo "Pair ${pair_index} failed; see ${left_log} and ${right_log}" >&2
        exit 2
    fi
    /home/niko/miniconda3/envs/photomaker_NS/bin/python \
        "${HERE}/export_4k_results.py" >>"${STATUS}"
done

echo "Completed: $(date -u '+%Y-%m-%dT%H:%M:%SZ')" >>"${STATUS}"
