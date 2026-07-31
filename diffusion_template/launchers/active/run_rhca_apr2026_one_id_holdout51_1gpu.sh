#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONFIG_NAME="one_id_rhca_apr2026_replay_holdout51"
export RUN_NAME="${RUN_NAME:-rhca_apr2026_one_id_holdout51_4k}"
export COMET_PROJECT="${COMET_PROJECT:-rsrch-jul}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-2}"  # 2 × 2,000 steps = 4,000 total

# Keep the historical RHCA architecture and validation contract, but exclude
# validation reference 51.jpg from both target and training-reference sampling.
exec bash "${SCRIPT_DIR}/run_rhca_apr2026_one_id_1gpu.sh" "$@"
