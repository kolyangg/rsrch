#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_NAME="${RUN_NAME:-rhca_apr2026_cosmic_large_one_id_faceonly_noca_4k}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-2}"  # 2 × 2,000 steps = 4,000 total
export COMET_PROJECT="${COMET_PROJECT:-rsrch-jul}"

# 24 Jul 2026 - Isolate branched cross-attention without changing the
# historical self-attention projection scheme or replay launcher.
exec bash "${SCRIPT_DIR}/run_rhca_apr2026_cosmic_large_one_id_faceonly_8k_1gpu.sh" \
  "disable_branched_ca=true" \
  "train_branched_ca_lora=false" \
  "model.train_branched_ca_lora=false" \
  "$@"
