#!/usr/bin/env bash
# Overnight ablation matrix: runs N7 -> N8 -> N9 one after another.
# Design & rationale: debug_04Jul/7Jul_experiments_analysis.md  (results feed 04Jul_findings.md §9).
#
# Each experiment self-stops after 3000 steps (trainer.n_epochs x trainer.epoch_len), exits, and
# the next starts. A failure in one run is logged and the master CONTINUES to the next (so a single
# crash doesn't waste the night). Per-run stdout/stderr -> serv_new_runs/logs/<run>_<ts>.log.
#
# Total wall-clock estimate on the 45 GB card (~2.5 s/step, ~15-20 min/val, CUDA async):
#   N7  3000 steps + 6 vals  ~= 3.5-4 h
#   N8  3000 steps + 3 vals  ~= 3-3.5 h
#   N9  3000 steps + 3 vals  ~= 3-3.5 h
#   -------------------------------------
#   total ~= 9.5-11 h   (fits an overnight; if a night is short, drop N9 or lower n_epochs)
#
# USAGE (from the repo root, so relative paths resolve):
#   cd /path/to/diffusion_template
#   tmux new -s overnight        # or: nohup ... &   (survives disconnects)
#   bash serv_new_runs/run_overnight_N7_N8_N9.sh
#
# Results land in saved/ba_nr_blend_N7, saved/ba_nr_blend_N8, saved/ba_nr_blend_N9
# (each with val_images/step_*, weights-epoch*.pth, config.yaml, info.log). Compare per-step
# id-sim across the three vs the step-0 baseline (0.40); see the plan doc for the read-out.

set -uo pipefail   # NB: no -e; we handle per-run failures ourselves and keep going.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
cd "${REPO}"

LOGDIR="${HERE}/logs"
mkdir -p "${LOGDIR}"

# The experiments, in order. Edit/reorder/comment-out lines to change the plan.
EXPERIMENTS=(
    "start_ba_nr_blend_vast_N7.sh"
    "start_ba_nr_blend_vast_N8.sh"
    "start_ba_nr_blend_vast_N9.sh"
)

ts() { date '+%Y-%m-%d %H:%M:%S'; }
MASTER_LOG="${LOGDIR}/overnight_master_$(date '+%Y%m%d_%H%M%S').log"

echo "[$(ts)] overnight matrix start; repo=${REPO}" | tee -a "${MASTER_LOG}"
echo "[$(ts)] experiments: ${EXPERIMENTS[*]}" | tee -a "${MASTER_LOG}"

overall=0
for exp in "${EXPERIMENTS[@]}"; do
    script="${HERE}/${exp}"
    stamp="$(date '+%Y%m%d_%H%M%S')"
    runlog="${LOGDIR}/${exp%.sh}_${stamp}.log"

    if [[ ! -f "${script}" ]]; then
        echo "[$(ts)] SKIP ${exp}: script not found" | tee -a "${MASTER_LOG}"
        overall=1
        continue
    fi

    echo "[$(ts)] >>> START ${exp}  (log: ${runlog})" | tee -a "${MASTER_LOG}"
    start_s=$(date +%s)

    # Run the experiment; capture its exit code without aborting the master.
    bash "${script}" >"${runlog}" 2>&1
    rc=$?

    dur=$(( $(date +%s) - start_s ))
    if [[ ${rc} -eq 0 ]]; then
        echo "[$(ts)] <<< DONE  ${exp}  rc=0  (${dur}s = $((dur/60)) min)" | tee -a "${MASTER_LOG}"
    else
        echo "[$(ts)] <<< FAIL  ${exp}  rc=${rc}  (${dur}s) — continuing to next; see ${runlog}" | tee -a "${MASTER_LOG}"
        overall=1
    fi
done

echo "[$(ts)] overnight matrix finished; overall_status=${overall}" | tee -a "${MASTER_LOG}"
exit "${overall}"
