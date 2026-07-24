#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OLD_QUEUE_PID="${1:?usage: continue_queue_after_full_replacement.sh OLD_QUEUE_PID}"
STATUS="${HERE}/scheduler_4k/REPLACEMENT_CONTINUATION_STATUS.md"

audit_pass_exists() {
    local architecture="$1"
    local audit
    while IFS= read -r audit; do
        run_dir="$(dirname -- "$(dirname -- "${audit}")")"
        if [[ "$(jq -r '.architecture_id // empty' "${run_dir}/run_manifest.json")" == "${architecture}" ]] \
            && [[ "$(jq -r '.status // empty' "${run_dir}/run_manifest.json")" == "completed" ]] \
            && [[ "$(jq -r '.status // empty' "${audit}")" == "PASS" ]]; then
            return 0
        fi
    done < <(
        find "${HERE}/experiments_4k" -mindepth 3 -maxdepth 3 \
            -path "*/report/comet_unity_audit.json" -type f 2>/dev/null
    )
    return 1
}

{
    echo "# Replacement continuation"
    echo
    echo "Armed: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo
    echo "Waiting for the retained CosmicLarge arm and replacement full18 OneID arm."
} >"${STATUS}"

while ! audit_pass_exists "L4_C2_large_projection_blend20" \
    || ! audit_pass_exists "L4_OF1_oneid_full18_projection_alt"; do
    sleep 20
done

echo "- $(date -u '+%Y-%m-%dT%H:%M:%SZ'): both retained/replacement arms passed final audits" \
    >>"${STATUS}"
if kill -0 "${OLD_QUEUE_PID}" 2>/dev/null; then
    kill -TERM "${OLD_QUEUE_PID}"
fi
echo "- $(date -u '+%Y-%m-%dT%H:%M:%SZ'): retired old pair-level queue" \
    >>"${STATUS}"

SKIP_PRIORITY_2=1 "${HERE}/schedule_4k_resume_missing.sh"
