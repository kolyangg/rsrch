#!/usr/bin/env bash
# Wait for the running full-validation batch (run_full_validation.sh) to finish, then start the
# N18 combo-10k training. Does NOT interrupt the validation. Run it in a SEPARATE terminal with the
# venv active; nohup it so it survives disconnects:
#
#   cd /workspace/rsrch/diffusion_template
#   nohup bash serv_new_runs/chain_after_validation.sh > serv_new_runs/logs/chain_N18.log 2>&1 &
#   echo "chain launcher PID $!"
#
# How it detects completion: the validation MASTER process (run_full_validation.sh) stays alive for
# the WHOLE batch — it runs each infer.py + metrics step synchronously — so watching that process is
# robust to the CPU-only metrics gaps between runs (a plain GPU-idle check would fire during those).
# This launcher's own cmdline is `bash chain_after_validation.sh`, so the pgrep pattern below does
# not match it.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
cd "${REPO}"
mkdir -p "${HERE}/logs"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[chain] $(ts) waiting for run_full_validation.sh to finish..."
sleep 10   # let this launcher settle before polling
while pgrep -f run_full_validation.sh >/dev/null 2>&1; do
  sleep 60
done
echo "[chain] $(ts) validation batch finished; confirming GPU is free..."
free=0
while true; do
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q "[0-9]"; then
    free=0
  else
    free=$((free + 1)); [ "${free}" -ge 3 ] && break
  fi
  sleep 20
done

echo "[chain] $(ts) launching N18 (combo 10k)"
bash "${HERE}/start_ba_combo10k_vast_N18.sh"
echo "[chain] $(ts) N18 exited with code $?"
