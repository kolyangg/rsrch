#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CURRENT_QUEUE_PID="${1:-}"
SCHEDULER_DIR="${HERE}/scheduler_4k"
mkdir -p "${SCHEDULER_DIR}"
STATUS="${SCHEDULER_DIR}/QUEUE_ALL_STATUS.md"

# Highest-priority projection/loss/value arms are already owned by the live
# schedule_4k_queue.sh. This continuation starts only after that parent exits.
PAIRS=(
    "L4_O4_oneid_projection_schedule|E12_projection_split_schedule|one_id_nm0005092_subset8_distinct|0 L4_C4_large_projection_schedule|E12_projection_split_schedule|cosmic_large_id00081|1"
    "L4_O5_oneid_active_up_blend20|E11_active_up_blended20|one_id_nm0005092_subset8_distinct|0 L4_C5_large_active_up_blend20|E11_active_up_blended20|cosmic_large_id00081|1"
    "L4_O6_oneid_active_up_schedule|E07_schedule_matched_up|one_id_nm0005092_subset8_distinct|0 L4_C6_large_active_up_schedule|E07_schedule_matched_up|cosmic_large_id00081|1"
    "L4_O7_oneid_noise_damped|E04_noise_damped|one_id_nm0005092_subset8_distinct|0 L4_C7_large_noise_damped|E04_noise_damped|cosmic_large_id00081|1"
    "L4_O8_oneid_all_blend20|E05_blended20|one_id_nm0005092_subset8_distinct|0 L4_C8_large_all_blend20|E05_blended20|cosmic_large_id00081|1"
    "L4_O9_oneid_projection_teacher20|E14_projection_split_pm_teacher20|one_id_nm0005092_subset8_distinct|0 L4_C9_large_projection_teacher20|E14_projection_split_pm_teacher20|cosmic_large_id00081|1"
    "L4_O10_oneid_control|E00_control|one_id_nm0005092_subset8_distinct|0 L4_C10_large_control|E00_control|cosmic_large_id00081|1"
    "L4_O11_oneid_active_up|E01_active_up|one_id_nm0005092_subset8_distinct|0 L4_C11_large_active_up|E01_active_up|cosmic_large_id00081|1"
    "L4_O12_oneid_up1_detail|E02_up1_detail|one_id_nm0005092_subset8_distinct|0 L4_C12_large_up1_detail|E02_up1_detail|cosmic_large_id00081|1"
    "L4_O13_oneid_staged_up1_up0|E03_staged_up1_up0|one_id_nm0005092_subset8_distinct|0 L4_C13_large_staged_up1_up0|E03_staged_up1_up0|cosmic_large_id00081|1"
)

{
    echo "# Complete 4k queue status"
    echo
    echo "Continuation armed: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo
    echo "Waiting for initial six-run queue PID: \`${CURRENT_QUEUE_PID:-not supplied}\`."
} >"${STATUS}"

if [[ -n "${CURRENT_QUEUE_PID}" ]]; then
    while kill -0 "${CURRENT_QUEUE_PID}" 2>/dev/null; do
        sleep 20
    done
fi
{
    echo
    echo "Initial queue released continuation: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
} >>"${STATUS}"

pair_index=3
for pair in "${PAIRS[@]}"; do
    pair_index=$((pair_index + 1))
    read -r left_record right_record <<<"${pair}"
    IFS='|' read -r left_id left_base left_dataset left_slot <<<"${left_record}"
    IFS='|' read -r right_id right_base right_dataset right_slot <<<"${right_record}"
    left_log="${SCHEDULER_DIR}/${left_id}.log"
    right_log="${SCHEDULER_DIR}/${right_id}.log"

    "${HERE}/run_4k_matrix_arm.sh" \
        "${left_id}" "${left_base}" "${left_dataset}" "${left_slot}" \
        >"${left_log}" 2>&1 &
    left_pid=$!
    "${HERE}/run_4k_matrix_arm.sh" \
        "${right_id}" "${right_base}" "${right_dataset}" "${right_slot}" \
        >"${right_log}" 2>&1 &
    right_pid=$!
    {
        echo
        echo "## Priority pair ${pair_index}"
        echo
        echo "- $(date -u '+%Y-%m-%dT%H:%M:%SZ'): started \`${left_id}\` from \`${left_base}\` PID ${left_pid}"
        echo "- $(date -u '+%Y-%m-%dT%H:%M:%SZ'): started \`${right_id}\` from \`${right_base}\` PID ${right_pid}"
    } >>"${STATUS}"

    set +e
    wait "${left_pid}"
    left_status=$?
    wait "${right_pid}"
    right_status=$?
    set -e
    echo "- $(date -u '+%Y-%m-%dT%H:%M:%SZ'): pair finished; statuses ${left_status}/${right_status}" \
        >>"${STATUS}"
    if (( left_status != 0 || right_status != 0 )); then
        echo "Priority pair ${pair_index} failed; see ${left_log} and ${right_log}" >&2
        exit 2
    fi
    /home/niko/miniconda3/envs/photomaker_NS/bin/python \
        "${HERE}/export_4k_results.py" >>"${STATUS}"
done

echo "Completed: $(date -u '+%Y-%m-%dT%H:%M:%SZ')" >>"${STATUS}"
