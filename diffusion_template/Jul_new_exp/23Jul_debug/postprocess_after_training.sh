#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd -- "${HERE}/../.." && pwd)"
ENV_BIN="/home/niko/miniconda3/envs/photomaker_NS/bin"
WATCH_PID="${1:?usage: postprocess_after_training.sh WATCH_PID ARCHITECTURE [CONTROL_SOURCE]}"
ARCHITECTURE="${2:?usage: postprocess_after_training.sh WATCH_PID ARCHITECTURE [CONTROL_SOURCE]}"
CONTROL_SOURCE="${3:-}"
VALIDATION_LOCK="${HERE}/scheduler/gpu_validation.lock"

timestamp() {
    date -u '+%Y-%m-%dT%H:%M:%SZ'
}

echo "$(timestamp) waiting to postprocess ${ARCHITECTURE} after PID ${WATCH_PID}"
while kill -0 "${WATCH_PID}" 2>/dev/null; do
    sleep 10
done

RUN_DIR="$(
    "${ENV_BIN}/python" - "${HERE}" "${ARCHITECTURE}" <<'PY'
import json
import sys
from pathlib import Path

here = Path(sys.argv[1])
architecture = sys.argv[2]
matches = []
for manifest_path in (here / "experiments").glob("*/run_manifest.json"):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("architecture_id") == architecture:
        matches.append((manifest_path.stat().st_mtime, manifest_path.parent, manifest))
if not matches:
    raise SystemExit(f"No experiment manifest found for {architecture}")
_, run_dir, manifest = max(matches)
if manifest.get("status") != "completed":
    raise SystemExit(
        f"Latest {architecture} run is not complete: "
        f"{manifest.get('status')} at {run_dir}"
    )
print(run_dir)
PY
)"
echo "$(timestamp) resolved completed run ${RUN_DIR}"

export PATH="${ENV_BIN}:${PATH}"
export CONDA_PREFIX="/home/niko/miniconda3/envs/photomaker_NS"
export CONDA_DEFAULT_ENV="photomaker_NS"
source "${PROJECT}/setup/env_snapshot_photomaker_NS/activate_runtime_photomaker_NS.sh"
export PYTHONPATH="${HERE}:${PROJECT}:${PYTHONPATH:-}"

"${ENV_BIN}/python" "${HERE}/checkpoint_diagnostics.py" "${RUN_DIR}"

if [[ -n "${CONTROL_SOURCE}" ]]; then
    if [[ "${CONTROL_SOURCE}" == architecture:* ]]; then
        SOURCE_ARCHITECTURE="${CONTROL_SOURCE#architecture:}"
        CONTROL_SOURCE="$(
            "${ENV_BIN}/python" - "${HERE}" "${SOURCE_ARCHITECTURE}" <<'PY'
import json
import sys
from pathlib import Path

here = Path(sys.argv[1])
architecture = sys.argv[2]
matches = []
for manifest_path in (here / "experiments").glob("*/run_manifest.json"):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("architecture_id") == architecture
        and manifest.get("status") == "completed"
    ):
        matches.append((manifest_path.stat().st_mtime, manifest_path.parent))
if not matches:
    raise SystemExit(f"No completed control source found for {architecture}")
print(max(matches)[1])
PY
        )"
    fi
    CONTROL_WAIT=0
    while true; do
        PM_COUNT="$(
            find "${CONTROL_SOURCE}/validation/pmControl50/step_0000/outputs" \
                -type f -name '*.png' ! -name '*_mask.png' 2>/dev/null \
                | wc -l
        )"
        STEP0_COUNT="$(
            find "${CONTROL_SOURCE}/validation/canonical50/step_0000/outputs" \
                -type f -name '*.png' ! -name '*_mask.png' 2>/dev/null \
                | wc -l
        )"
        if (( PM_COUNT == 4 && STEP0_COUNT == 4 )); then
            break
        fi
        CONTROL_WAIT=$((CONTROL_WAIT + 1))
        if (( CONTROL_WAIT >= 360 )); then
            echo "Timed out waiting for shared controls in ${CONTROL_SOURCE}" >&2
            exit 2
        fi
        echo "$(timestamp) waiting for shared controls in ${CONTROL_SOURCE}"
        sleep 10
    done
    "${ENV_BIN}/python" "${HERE}/materialize_shared_step0_controls.py" \
        "${RUN_DIR}" --source-run-dir "${CONTROL_SOURCE}"
    STEPS="200,400,600"
    MODES="canonical50"
else
    STEPS="0,200,400,600"
    MODES="canonical50,pmControl50"
fi

while ! mkdir "${VALIDATION_LOCK}" 2>/dev/null; do
    echo "$(timestamp) waiting for experiment validation lock"
    sleep 10
done
release_lock() {
    rmdir "${VALIDATION_LOCK}" 2>/dev/null || true
}
trap release_lock EXIT

LOW_COUNT=0
while true; do
    PRODUCTION_PID="$(
        pgrep -f '/home/niko/miniconda3/envs/photomaker_NS/bin/python -u train.py.*writer.run_name=ba_N3a_new1_1gpu' \
            | head -n 1 || true
    )"
    if [[ -z "${PRODUCTION_PID}" ]]; then
        break
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
        LOW_COUNT=$((LOW_COUNT + 1))
        if (( LOW_COUNT >= 3 )); then
            break
        fi
    else
        LOW_COUNT=0
    fi
    sleep 10
done

echo "$(timestamp) validating ${ARCHITECTURE}; steps=${STEPS}; modes=${MODES}"
"${HERE}/run_validation_suite.sh" "${RUN_DIR}" \
    --steps "${STEPS}" --modes "${MODES}"
release_lock
trap - EXIT

"${ENV_BIN}/python" "${HERE}/summarize_run.py" "${RUN_DIR}"
"${ENV_BIN}/python" "${HERE}/visualize_pm_masks.py" \
    --run-dir "${RUN_DIR}" \
    --output-dir "${RUN_DIR}/report/pm_bbox_debug"
"${ENV_BIN}/python" "${HERE}/upload_report_to_comet.py" "${RUN_DIR}"
echo "$(timestamp) completed report for ${ARCHITECTURE}: ${RUN_DIR}"
