#!/usr/bin/env bash
# Overnight batch: N14 (combo, primary) FIRST, then N15 -> N16. All 6000 steps.
# Design & rationale: debug_04Jul/7Jul_experiments_analysis.md §7.
#
# N13  identity loss   (objective change; the only lever likely to beat the step-0 baseline 0.40)
# N10  co-adaptation   (non_ba_train=true — train base LoRA + BA together)
# N11  SA-only         (train_branched_ca_lora=false — freeze the branched cross-attn)
# N12  id_embeds       (face_embed_strategy=id_embeds — ID features into the face-branch CA)
#
# Each experiment self-stops after 3000 steps (trainer.n_epochs x trainer.epoch_len), exits, and
# the next starts. A failure in one run is logged and the master CONTINUES to the next (so a single
# crash doesn't waste the night). Per-run stdout/stderr -> serv_new_runs/logs/<run>_<ts>.log.
#
# Total wall-clock on the 45 GB card (~2.5 s/step, ~15-20 min/val; N13 a bit slower due to the ID
# VAE decode on gated steps):
#   N13 ~= 3.5-4 h     N10 ~= 3-3.5 h     N11 ~= 3-3.5 h     N12 ~= 3-3.5 h
#   total ~= 12.5-14.5 h. If the night is short, comment out the tail (e.g. N12, or N11+N12) in the
#   EXPERIMENTS list below — N13 (the flagship) always runs first.
#
# USAGE (from the repo root, so relative paths resolve):
#   cd /path/to/diffusion_template
#   tmux new -s overnight        # or nohup ... &  (survives disconnects)
#   bash serv_new_runs/run_overnight_N14_N15_N16.sh
#
# Results in saved/{ba_combo_N14, ba_saonly6k_N15, ba_idloss6k_N16}. Score with
# scripts/idsim_report.py; for N13 also watch train/id_loss in Comet (should trend DOWN). Target to
# beat: step-0 0.40 (untrained), N6 0.297 (best trained so far).

set -uo pipefail   # NB: no -e; we handle per-run failures ourselves and keep going.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
cd "${REPO}"

LOGDIR="${HERE}/logs"
mkdir -p "${LOGDIR}"

# The experiments, in order. Comment out any line to skip it. N13 (flagship) is first.
EXPERIMENTS=(
    "start_ba_combo_vast_N14.sh"
    "start_ba_saonly6k_vast_N15.sh"
    "start_ba_idloss6k_vast_N16.sh"
)

ts() { date '+%Y-%m-%d %H:%M:%S'; }
MASTER_LOG="${LOGDIR}/overnight_master_$(date '+%Y%m%d_%H%M%S').log"

echo "[$(ts)] overnight batch start; repo=${REPO}" | tee -a "${MASTER_LOG}"
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

echo "[$(ts)] overnight batch finished; overall_status=${overall}" | tee -a "${MASTER_LOG}"
exit "${overall}"
