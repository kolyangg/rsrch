#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_NAME="${RUN_NAME:-rhca_apr2026_cosmic_large_one_id_faceonly_noca_refonly_4k}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-2}"  # 2 × 2,000 steps = 4,000 total
export COMET_PROJECT="${COMET_PROJECT:-rsrch-jul}"

# 24 Jul 2026 - Freeze target/noise Q/K/V copies while retaining the explicit
# reference K/V path; this arm must differ from the CA-off arm by one variable.
exec bash "${SCRIPT_DIR}/run_rhca_apr2026_cosmic_large_one_id_faceonly_noca_4k_1gpu.sh" \
  "branched_attn_weight_mode=ref_only" \
  "model.branched_attn_weight_mode=ref_only" \
  "$@"
