#!/usr/bin/env bash
# Wait for the running N17 long run (ba_longrun_N17) to finish, then start N19 (CA-trainable long
# run). Does NOT interrupt N17. Run in a SEPARATE terminal with the venv active, nohup'd:
#
#   cd /workspace/rsrch/diffusion_template
#   nohup bash serv_new_runs/chain_after_N17.sh > serv_new_runs/logs/chain_N19.log 2>&1 &
#   echo "chain launcher PID $!"
#
# It watches the N17 training PROCESS (its cmdline contains writer.run_name=ba_longrun_N17). This
# launcher's own cmdline is `bash chain_after_N17.sh`, so the pgrep pattern does not match it.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
cd "${REPO}"
mkdir -p "${HERE}/logs"
ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[chain] $(ts) waiting for N17 (ba_longrun_N17) to finish..."
sleep 10
while pgrep -f ba_longrun_N17 >/dev/null 2>&1; do sleep 60; done
echo "[chain] $(ts) N17 finished; confirming GPU is free..."
free=0
while true; do
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q "[0-9]"; then
    free=0
  else
    free=$((free + 1)); [ "${free}" -ge 3 ] && break
  fi
  sleep 20
done
echo "[chain] $(ts) launching N19 (CA-trainable 15k)"
bash "${HERE}/start_ba_catrain_vast_N19.sh"
echo "[chain] $(ts) N19 exited with code $?"
