#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_NAME="${RUN_NAME:-rhca_apr2026_cosmic_large_one_id_faceonly_8k}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-16}"  # 16 × 500 steps = 8,000 total
export COMET_PROJECT="${COMET_PROJECT:-rsrch-jul}"

# Clean face-only run: preserve the RHCA architecture and Cosmic dataset while
# applying masked face MSE on every optimizer step.
exec bash "${SCRIPT_DIR}/run_rhca_apr2026_cosmic_large_one_id_1gpu.sh" \
  "continue_run=false" \
  "saved_checkpoint=null" \
  "trainer.resume_from=null" \
  "trainer.masked_loss_step=1" \
  "$@"
