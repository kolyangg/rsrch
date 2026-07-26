#!/usr/bin/env bash
set -euo pipefail

export RUN_ID="rhca_cosmic_full_crop20_posefirst_par100_4k_r2_full96_par100"
export VALIDATION_SOURCE_RUN="rhca_cosmic_full_crop20_posefirst_par100_4k_r2"
export VALIDATION_SOURCE_COMET_KEY="e6cfd6b676ba474fad5f97824ec3d37d"
export CANONICAL_FULL96_RUN="rhca_cosmic_full_crop20_posefirst_4k_fast_r3_full96_par100"
export POSE_ADAPT_RATIO="1.0"
export FULL96_ACTIVE_LAUNCHER="launchers/active/run_rhca_cosmic_pose_adapt_full96_eval_1gpu.sh"

exec bash "$(dirname "${BASH_SOURCE[0]}")/../_lib/start_cosmic_full96_fast_env.sh"
